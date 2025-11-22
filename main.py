import asyncio
from typing import Dict, Optional, List, Any
from pathlib import Path
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import os


from ollama_agent import create_agent as create_ollama_agent
from agent.memory import ValkeyMemoryStore

app = FastAPI(
    title="AI Act Knowledge Graph Chatbot",
    description="Chatbot that uses LangChain tools to explore the EU AI Act knowledge graph with Langfuse observability.",
)

store = ValkeyMemoryStore(valkey_uri=os.environ.get("VALKEY_URI", "localhost"))


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the EU AI Act.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier used to group related questions.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional key-value metadata forwarded to Langfuse traces.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chatbot answer grounded in the AI Act knowledge graph.")
    observation_log: List[Any] = Field(
        default=None,
        description="Log of agent steps",
    )


# Serve static site (your matrix UI)
static_dir = Path(__file__).parent / "frontend" /"static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
template_dir = Path(__file__).parent / "frontend" / "templates"
templates = Jinja2Templates(directory=template_dir)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Serve index.html so users can open http://localhost:8080/
    return templates.TemplateResponse(
        request=request, name="chat.html"
    )

@app.get("/explain", response_class=HTMLResponse)
def explain(request: Request):
    return templates.TemplateResponse(
        request=request, name="explain.html"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat2(request: ChatRequest) -> ChatResponse:
    metadata = request.metadata or {}
    if request.session_id:
        metadata = {**metadata, "session_id": request.session_id}
    
    memory = store.get_or_create(request.session_id)
    ollama_agent = create_ollama_agent(memory=memory)

    try:
        answer, observation_log = await asyncio.to_thread(
            ollama_agent.chat,
            question=request.question,
            metadata=metadata if metadata else None
        )
        store.save(request.session_id, memory)
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ollama agent execution failed: {exc}") from exc

    return ChatResponse(answer=answer, observation_log=observation_log)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}
