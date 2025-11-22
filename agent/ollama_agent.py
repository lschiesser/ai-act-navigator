"""
Custom ReAct-style agent that talks directly to Ollama and can call local tools.

The agent uses Ollama's function calling interface to decide when to run one of
the available knowledge-graph tools and iterates until the model returns a
final answer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import ollama
import sys, os

#from tools.graph_traversal import GraphTraversal
from tools.graph_traversal import TraversalTool
from tools.node_retrieval import NodeRetrievalTool
from tools.single_node_retriever import SingleNodeRetriever

from agent.memory import ShortTermMemory, ValkeyMemoryStore
from utils.utils import _json_fallback


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def to_ollama_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReActAgent:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_tool_iterations: int = 6,
        memory: Optional[ShortTermMemory | ValkeyMemoryStore] = None,
    ) -> None:
        self.model = model or os.getenv("MODEL", "qwen2.5:32b")
        self.host = host or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)
        self.max_tool_iterations = max_tool_iterations
        self.memory = memory

        self._node_retrieval = NodeRetrievalTool(host=self.host)
        self._graph_traversal = TraversalTool()
        self._single_node_retriever = SingleNodeRetriever()

        self._tools: Dict[str, ToolDescriptor] = {
            "node_retrieval": ToolDescriptor(
                name="node_retrieval",
                description="Search the AI Act knowledge graph for relevant passages.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Natural language query describing the information to retrieve.",
                        }
                    },
                    "required": ["query_text"],
                },
                handler=self._run_node_retrieval,
            ),
            "graph_traversal": ToolDescriptor(
                name="graph_traversal",
                description="Retrieves a node's context (e.g., subparagraphs) and references from the knowledge graph, assembling hierarchical text and metadata.",
                parameters={
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "Identifier of the node to traverse.",
                        }
                    },
                    "required": ["node_id"],
                },
                handler=self._run_graph_traversal,
            ),
            "single_node_retriever": ToolDescriptor(
                name="single_node_retriever",
                description="Retrieves a single node without its context from the knowledge graph using its unique ID. The IDs follow the AI Act's hierarchical naming scheme, i.e. Art.<number> for Article, Annex.<roman numeral> for Annexes, Rec.<number> for Recitals and Chapt<Roman numeral> for Chapters.",
                parameters={
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "Identifier of the node to retrieve.",
                        }
                    },
                    "required": ["node_id"],
                },
                handler=self._run_single_node_retriever,
            )
        }

        self._tool_specs: List[Dict[str, Any]] = [
            tool.to_ollama_spec() for tool in self._tools.values()
        ]

        self._system_prompt = (
            "You are an assistant that answers questions about the EU AI Act. "
            "You have access to the functions listed in the tools section and should use to ground your answer in the knowledge graph."
            "Base your answers only on the output from the tools, otherwise answer that the AI Act does not include enough information to answer the question properly."
            "Follow the ReAct pattern: think about the problem, call tools with well-formed JSON arguments when needed, "
            "and continue reasoning based on their observations. "
            "If you are using the single_node_retriever, it might be helpful to then run the graph_traversal function."
            "When you are confident in your response, reply with a concise answer prefixed with 'FINAL:'."
        )

    def close(self) -> None:
        self._node_retrieval.close()
        if hasattr(self._graph_traversal, "close"):
            self._graph_traversal.close()  # type: ignore[attr-defined]

    def invoke(
        self,
        inputs: Dict[str, Any],
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        question = inputs.get("input") or inputs.get("question")
        if not question:
            raise ValueError("invoke expects an 'input' or 'question' key with user text.")
        answer = self.chat(question=question, metadata=(config or {}).get("metadata"))
        return {"output": answer}

    def chat(
        self,
        *,
        question: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages: List[Dict[str, Any]] = self.memory.as_ollama_messages(
            self._system_prompt,
            metadata=metadata,
        )
        observation_log = []
        messages.append({"role": "user", "content": question})
        self.memory.add_user(question)

        for i in range(self.max_tool_iterations):
            print(i)
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=self._tool_specs,
            )
            assistant_message = response.get("message", {})
            self.memory.add_assistant(assistant_message.get("content", ""), phase="reasoning")
            tool_calls = assistant_message.get("tool_calls", [])

            if tool_calls:
                messages.append(assistant_message)
                for call in tool_calls:
                    tool_name = call.get("function", {}).get("name")
                    arguments = call.get("function", {}).get("arguments", {})
                    tool_result_content, summary = self._execute_tool(tool_name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result_content, ensure_ascii=False, default=_json_fallback),
                        }
                    )
                    self.memory.add_tool(tool_name, arguments, summary)
                    observation_log.append((call.get("function", {}), tool_result_content))
                continue

            content = assistant_message.get("content", "")
            answer = self._extract_final(content)    
            observation_log.append(("agent_step", assistant_message.get("content", "")))
            if answer:
                messages.append(assistant_message)
                self.memory.add_assistant(answer, phase="final")
                return answer, observation_log
            messages.append(assistant_message)
            self.memory.add_assistant(content, phase="final")
            return content, observation_log

        raise RuntimeError("Exceeded maximum number of tool iterations without a final answer.")

    def _execute_tool(self, tool_name: Optional[str], arguments: Any) -> str:
        if not tool_name or tool_name not in self._tools:
            return json.dumps(
                {"error": f"Unknown tool '{tool_name}'."},
                ensure_ascii=False,
            )

        tool = self._tools[tool_name]

        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"input": arguments}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            parsed_arguments = {"input": arguments}

        try:
            result, summary = tool.handler(**parsed_arguments)
        except TypeError as exc:
            result, summary = {"error": f"Invalid arguments for {tool_name}: {exc}"}, f"Error: Invalid arguments for {tool_name}: {exc}"
        except Exception as exc:  # pylint: disable=broad-except
            result, summary = {"error": f"Tool {tool_name} raised an exception: {exc}"}, f"Error: Tool {tool_name} raised an exception: {exc}"

        return result, summary

    def _run_node_retrieval(self, query_text: str, **_: Any) -> Dict[str, Any]:
        result = self._node_retrieval.retrieve(query_text=query_text)
        summary = [item["id"] for item in result["results"]]
        return result, f"Graph executed node retrieval with query: {query_text}. The query returned these nodes {', '.join(summary)}."

    def _run_graph_traversal(self, node_id: str, **_: Any) -> Dict[str, Any]:
        return self._graph_traversal.traverse(start_node_id=node_id), f"Agent executed graph traversal on {node_id}"
    
    def _run_single_node_retriever(self, node_id: str, **_:Any) -> Dict[str,Any]:
        return self._single_node_retriever.get_node(node_id=node_id), f"Agent retrieved node with id: {node_id} "
    
    @staticmethod
    def _extract_final(text: str) -> Optional[str]:
        import re
        m = re.search(r'FINAL:\s*(.*)$', text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None


def create_agent(
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    max_tool_iterations: int = 6,
    memory = None
) -> ReActAgent:
    return ReActAgent(model=model, host=host, max_tool_iterations=max_tool_iterations, memory=memory)


__all__ = ["ReActAgent", "create_agent"]
