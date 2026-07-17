import json
import sys
import uuid
import time
from typing import cast
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.state import MultiAgentState

# -----------------------------------------------------------------------
# Import the existing compiled rag_agent graph
# -----------------------------------------------------------------------
from agents.rag_agent import graph as rag_agent_graph

# -----------------------------------------------------------------------
# Helper: extract structured tool output from message history
# -----------------------------------------------------------------------
def extract_tool_output(messages: list) -> tuple[dict, str | None]:
    """
    Finds the last ToolMessage in the message list and returns
    (parsed_output, tool_name).

    tool_name tells the caller WHICH tool actually ran (e.g. "rag_search",
    "web_search") — this is the key fix. Previously this function only
    checked whether JSON parsing succeeded, which meant "no tool was
    called" and "web_search was called (raw string, not JSON)" both
    silently collapsed into the same empty-dict fallback.

    Returns ({}, None) if no ToolMessage exists at all — meaning the
    ReAct agent answered directly with no tool call (general knowledge).
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None)
            content = message.content
            if not isinstance(content, str):
                return {}, tool_name

            if tool_name == "rag_search":
                try:
                    return json.loads(content), tool_name
                except (ValueError, SyntaxError):
                    # rag_search's structured JSON failed to parse —
                    # genuine error, not the general-knowledge case
                    return {}, tool_name

            # web_search (or any other future tool): content is raw text,
            # not JSON. Don't try to json.loads it — just pass it through.
            return {"_raw_content": content}, tool_name

    # No ToolMessage found at all — agent answered directly, no tool call
    return {}, None

# -----------------------------------------------------------------------
# Research Node
# -----------------------------------------------------------------------
def research_node(state: MultiAgentState) -> dict:
    """
    Wraps the existing rag_agent graph as a research node.
    Invokes the agent, extracts structured tool output,
    maps fields to MultiAgentState, and logs the decision.
    """

    # Added - start timer for latency measurement
    node_start = time.time()
    
    query = state["query"]

    # Run the existing ReAct agent
    config: RunnableConfig = {
        "recursion_limit": 10,
        "configurable": {"thread_id": str(uuid.uuid4())}  # unique thread for each invocation
    }
    result = rag_agent_graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    )

    messages = result.get("messages", [])

    # for message in messages:
    #     if isinstance(message, ToolMessage):
    #         print("RAW TOOL CONTENT REPR:")
    #         print(repr(message.content[:500]))
    #         break


    # Identify which tool actually ran (or if none did at all)
    tool_output, tool_name = extract_tool_output(messages)
    node_latency_ms = round((time.time() - node_start) * 1000, 2)

    # -----------------------------------------------------------------
    # PATH 1 — rag_search ran, structured JSON parsed successfully
    # -----------------------------------------------------------------
    if tool_name == "rag_search" and tool_output:
        research_context = tool_output.get("context", "")
        sources = tool_output.get("sources", [])
        confidence = tool_output.get("confidence", "low")
        action = tool_output.get("action", "retry_or_abstain")
        answerable = tool_output.get("answerable", False)
        retrieval_strategy = tool_output.get("retrieval_strategy", "unknown")

        log_entry = (
            f"[ResearchAgent] Query: '{query}' | source_type: rag | "
            f"Confidence: {confidence} | Action: {action} | "
            f"Answerable: {answerable} | Strategy: {retrieval_strategy} | "
            f"Sources: {len(sources)} | Latency: {node_latency_ms} ms"
        )
        return {
            "research_context": research_context,
            "sources": sources,
            "confidence": confidence,
            "action": action,
            "answerable": answerable,
            "retrieval_strategy": retrieval_strategy,
            "source_type": "rag",
            "node_latencies": {"research_node": node_latency_ms},
            "agent_log": [log_entry],
        }

    # -----------------------------------------------------------------
    # PATH 2 — web_search ran. Its output is a raw stringified dict
    # (see tavily_tool.py: `return str(result)`), not JSON — so we
    # use it as-is as the research_context, we don't try to parse it.
    # -----------------------------------------------------------------
    if tool_name == "web_search":
        raw_content = tool_output.get("_raw_content", "")
        log_entry = (
            f"[ResearchAgent] Query: '{query}' | source_type: web_search | "
            f"Latency: {node_latency_ms} ms"
        )
        return {
            "research_context": raw_content,
            "sources": [],   # KNOWN GAP: Tavily result isn't parsed for
                              # individual URLs here — raw text only.
                              # Fine for now, flag if source-level
                              # citation is needed later.
            "confidence": "high",
            "action": "generate",
            "answerable": True,
            "retrieval_strategy": "web_search",
            "source_type": "web_search",
            "node_latencies": {"research_node": node_latency_ms},
            "agent_log": [log_entry],
        }

    # -----------------------------------------------------------------
    # PATH 3 — no ToolMessage at all. The ReAct agent answered directly
    # from its own knowledge (rag_agent.py's system prompt rule 3:
    # stable facts, math, definitions, etc.). There is no external
    # context to verify against — final_answer is set HERE, directly,
    # since writer_node will just pass it through unchanged.
    # -----------------------------------------------------------------
    final_message = messages[-1].content if messages else ""
    log_entry = (
        f"[ResearchAgent] Query: '{query}' | source_type: general_knowledge | "
        f"No tool called — agent answered directly | Latency: {node_latency_ms} ms"
    )
    return {
        "research_context": "",
        "sources": [],
        "confidence": "high",
        "action": "generate",
        "answerable": True,
        "retrieval_strategy": "none",
        "source_type": "general_knowledge",
        "final_answer": final_message if isinstance(final_message, str) else str(final_message),
        "node_latencies": {"research_node": node_latency_ms},
        "agent_log": [log_entry],
    }

if __name__ == "__main__":
    # Test the research_node function
    test_state = cast(MultiAgentState, {
        "query": "What is the opposite of happy?"
    })

    result = research_node(test_state)
    print("=== Research Node Test Result ===")
    print(f"Research Context: {result['research_context'][:300]}...")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Action: {result['action']}")
    print(f"Answerable: {result['answerable']}")
    print(f"Retrieval Strategy: {result['retrieval_strategy']}")
    print(f"Node Latencies: {result['node_latencies']}")
    print(f"Source Type: {result['source_type']}")
    print(f"Abstained: {result.get('abstained', False)}")
    print(f"Agent Log: {result['agent_log']}")