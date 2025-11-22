# memory.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import time
import uuid
import threading
from ollama import Client
import json, redis, os
from utils.utils import _json_fallback


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Message:
    role: str                    # "user" | "assistant" | "tool" | "system"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class MemoryNote:
    text: str
    turn_ids: List[str]
    score: float = 0.0           # cached score for last retrieval pass
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# -----------------------------
# Utility: cosine similarity
# -----------------------------
def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def rough_token_count(text: str) -> int:
    # Very rough. Replace with a proper tokenizer if you need tighter budgeting.
    return max(1, len(text.split()))


# -----------------------------
# ShortTermMemory
# -----------------------------
class ShortTermMemory:
    """
    - buffer_max_turns: cap on raw turns we keep before summarizing/evicting
    - reflect_every: summarize/compress after this many user+assistant turns
    - use_embeddings: enable semantic recall via Ollama embeddings
    """
    def __init__(
        self,
        llm_model: str = "gemma3:4b",
        embed_model: Optional[str] = "embeddinggemma:latest",   # e.g., "nomic-embed-text"
        buffer_max_turns: int = 32,
        reflect_every: int = 6,
        alpha_relevance: float = 0.60,
        beta_recency: float = 0.35,
        gamma_role: float = 0.05,
    ):
        self.buffer: List[Message] = []
        self.notes: List[MemoryNote] = []
        self.ollama = Client(host=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.buffer_max_turns = buffer_max_turns
        self.reflect_every = reflect_every
        self.turn_counter = 0
        self.alpha = alpha_relevance
        self.beta = beta_recency
        self.gamma = gamma_role
        self._lock = threading.RLock()

        # Embedding cache: id -> vector
        self._emb_cache: Dict[str, List[float]] = {}

    # ---- Adding content ----
    def add_user(self, content: str, **meta) -> str:
        with self._lock:
            m = Message(role="user", content=content, meta=meta)
            self.buffer.append(m)
            self.turn_counter += 1
            self._maybe_reflect()
            return m.id

    def add_assistant(self, content: str, **meta) -> str:
        with self._lock:
            m = Message(role="assistant", content=content, meta=meta)
            self.buffer.append(m)
            self.turn_counter += 1
            self._maybe_reflect()
            return m.id

    def add_tool(self, name: str, input_args: Dict[str, Any], result: Any, **meta) -> str:
        with self._lock:
            content = f"[tool:{name}] args={input_args!r}\nresult={result!r}"
            m = Message(role="tool", content=content, meta={"tool": name, **meta})
            self.buffer.append(m)
            # tool steps don't count toward reflection cadence by default
            return m.id

    # ---- Reflection & summarization ----
    def _maybe_reflect(self):
        # Summarize older content if enough turns passed or buffer too large
        if self.turn_counter % self.reflect_every == 0 or len(self.buffer) > self.buffer_max_turns:
            self._reflect()

    def _reflect(self):
        """
        Compress the **older half** of the buffer into MemoryNotes,
        keeping the newer half verbatim.
        """
        if not self.buffer:
            return

        # Partition buffer: older (to summarize), newer (to keep)
        midpoint = max(1, len(self.buffer) // 2)
        older = self.buffer[:midpoint]
        newer = self.buffer[midpoint:]

        # Build a compact transcript to summarize
        transcript = []
        turn_ids = []
        for m in older:
            transcript.append(f"{m.role.upper()}: {m.content}")
            turn_ids.append(m.id)
        summary_input = "\n".join(transcript)

        # Ask the LLM to produce distilled notes
        system = (
            "You compress chat history into concise, factual bullet points. "
            "Capture tasks, constraints, decisions, tool outcomes, and open questions. "
            "Avoid fluff. Use neutral wording."
        )
        prompt = (
            "Condense the following conversation into 3–7 bullets. "
            "Prefer reusable facts and tasks that matter for the next 10 minutes.\n\n"
            f"{summary_input}\n\n"
            "Bullets:"
        )
        notes_text = self._llm(system, prompt)

        # Create a MemoryNote and shrink buffer
        self.notes.append(MemoryNote(text=notes_text.strip(), turn_ids=turn_ids))
        self.buffer = newer  # keep only newer half

        # Optional: de-duplicate similar notes (simple Jaccard on tokens)
        self._dedupe_notes()

    def _dedupe_notes(self, similarity_threshold: float = 0.9):
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb: return 0.0
            return len(sa & sb) / len(sa | sb)
        unique: List[MemoryNote] = []
        for n in self.notes:
            if all(jaccard(n.text, m.text) < similarity_threshold for m in unique):
                unique.append(n)
        self.notes = unique

    # ---- Retrieval ----
    def retrieve(self, query: str, token_budget: int = 1200) -> List[Tuple[str, str]]:
        """
        Returns a list of (source, text) pairs:
          source ∈ {"note","msg"}
        Priority = α*relevance + β*recency + γ*role_boost
        """
        now = time.time()

        candidates: List[Tuple[str, str, float]] = []  # (kind, text, score)

        # Prepare embeddings (optional)
        q_vec = self._embed_once("__query__", query) if self.embed_model else None

        # Score notes
        for n in self.notes:
            recency = self._recency_score(now - n.ts)
            rel = cosine(q_vec, self._embed_note(n)) if q_vec is not None else self._keyword_rel(query, n.text)
            score = self.alpha * rel + self.beta * recency
            n.score = score
            candidates.append(("note", n.text, score))

        # Score buffer messages (recent raw turns & tool traces)
        for m in self.buffer:
            role_boost = 0.1 if m.role in ("tool", "system") else 0.0
            recency = self._recency_score(now - m.ts)
            rel = cosine(q_vec, self._embed_msg(m)) if q_vec is not None else self._keyword_rel(query, m.content)
            score = self.alpha * rel + self.beta * recency + self.gamma * role_boost
            candidates.append(("msg", f"{m.role}: {m.content}", score))

        # Sort and pack under token budget
        candidates.sort(key=lambda x: x[2], reverse=True)

        packed: List[Tuple[str, str]] = []
        used = 0
        for kind, text, _ in candidates:
            tc = rough_token_count(text)
            if used + tc > token_budget:
                continue
            packed.append((kind, text))
            used += tc

        # Always add the very last user message if not already included
        last_user = next((m for m in reversed(self.buffer) if m.role == "user"), None)
        if last_user:
            snippet = ("msg", f"user: {last_user.content}")
            if snippet not in packed:
                if used + rough_token_count(last_user.content) <= token_budget:
                    packed.append(snippet)

        return packed

    # ---- Internals ----
    def _recency_score(self, age_seconds: float) -> float:
        # Fresh items ~1.0, decays over ~30 minutes
        half_life = 30 * 60
        return math.exp(-math.log(2) * (age_seconds / half_life))

    def _keyword_rel(self, query: str, text: str) -> float:
        q = set(query.lower().split())
        t = set(text.lower().split())
        if not q or not t: return 0.0
        return len(q & t) / len(q)

    def _embed_once(self, key: str, text: str) -> List[float]:
        if key in self._emb_cache:
            return self._emb_cache[key]
        if not self.embed_model:
            return []
        vec = self.ollama.embed(self.embed_model, [text])[0]
        self._emb_cache[key] = vec
        return vec

    def _embed_msg(self, m: Message) -> List[float]:
        if not self.embed_model:
            return []
        key = f"msg:{m.id}"
        if key in self._emb_cache:
            return self._emb_cache[key]
        vec = self.ollama.embed(self.embed_model, [m.content])[0]
        self._emb_cache[key] = vec
        return vec

    def _embed_note(self, n: MemoryNote) -> List[float]:
        if not self.embed_model:
            return []
        key = f"note:{n.id}"
        if key in self._emb_cache:
            return self._emb_cache[key]
        vec = self.ollama.embed(self.embed_model, [n.text])[0]
        self._emb_cache[key] = vec
        return vec

    def _llm(self, system: str, user: str) -> str:
        result = self.ollama.chat(
            model=self.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return result["message"]["content"]
    
    def as_ollama_messages(
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Any] = None,
        include_notes: bool = True,
        include_tools: bool = False,
        max_turns: Optional[int] = None,
        recent_only: bool = True,
        query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Convert memory into a structured message list for ollama.chat().

        Args:
            system_prompt: optional system instruction
            include_notes: if True, include MemoryNotes as system messages
            include_tools: if True, include tool traces as assistant messages
            max_turns: limit number of recent buffer messages
            recent_only: keep only the most recent turns if True
            query: optional query string for contextual retrieval (future extension)
        """
        messages: List[Dict[str, str]] = []

        # (1) Add global system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if metadata:
            serialized_metadata = json.dumps(metadata, ensure_ascii=False, default=_json_fallback)
            messages.append(
                {
                    "role": "system",
                    "content": f"Metadata for context: {serialized_metadata}",
                }
            )

        # (2) Optionally include summaries (MemoryNotes)
        if include_notes and self.notes:
            for note in self.notes[-3:]:  # keep last 3 summaries
                messages.append({
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{note.text.strip()}"
                })

        # (3) Add actual conversation history
        turns = self.buffer[-max_turns:] if max_turns else self.buffer
        for m in turns:
            if m.role == "tool" and not include_tools:
                continue
            messages.append({"role": m.role, "content": m.content})

        # Ensure chronological order
        messages.sort(key=lambda msg: msg.get("ts", 0))

        return messages


class SerializableMemory(ShortTermMemory):
    def to_dict(self):
        return {
            "buffer": [m.__dict__ for m in self.buffer],
            "notes": [n.__dict__ for n in self.notes],
        }

    @classmethod
    def from_dict(cls, data, **kwargs):
        mem = cls(**kwargs)
        mem.buffer = [Message(**m) for m in data.get("buffer", [])]
        mem.notes = [MemoryNote(**n) for n in data.get("notes", [])]
        return mem

    def dumps(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def loads(cls, s: str, **kwargs):
        return cls.from_dict(json.loads(s), **kwargs)

class ValkeyMemoryStore:
    """
    Session-based memory store using Valkey (drop-in Redis alternative).
    Stores each session_id -> serialized ShortTermMemory JSON.
    """
    def __init__(self, valkey_uri="redis://:@localhost:6379", ttl_seconds: int = 3600):
        self.db = redis.Redis.from_url(valkey_uri)
        self.ttl = ttl_seconds  # expire memory after inactivity

    def get_or_create(self, session_id: str) -> SerializableMemory:
        data = self.db.get(session_id)
        if data:
            return SerializableMemory.loads(data)
        return SerializableMemory()

    def save(self, session_id: str, memory: SerializableMemory):
        self.db.set(session_id, memory.dumps(), ex=self.ttl)