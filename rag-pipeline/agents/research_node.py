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
def extract_tool_output(messages: list) -> dict:
    """
    Finds the last ToolMessage in the message list and parses
    its content as a Python dict.
    Returns empty dict if no ToolMessage found or parsing fails.
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = message.content
            if not isinstance(content, str):
                return {}
            try:
                return json.loads(content)
            except (ValueError, SyntaxError):
              #ast.literal_eval fails on JSON because JSON uses "double quotes" for keys and values. Python literals use 'single quotes'. So it throws a ValueError, returns {}, and you hit the fallback.
                return {}    
        
    return {}

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


    # Extract structured output from ToolMessage
    tool_output = extract_tool_output(messages)

     # ADDED: stop timer before building return dict
    node_latency_ms = round((time.time() - node_start) * 1000, 2)

    # If no structured tool output found (e.g. web_search was used),
    # fall back to the final LLM message as raw context
    if not tool_output:
        final_message = messages[-1].content if messages else ""
        return {
            "research_context": final_message,
            "sources": [],
            "confidence": "low",
            "action": "generate_cautiously",
            "answerable": True,
            "retrieval_strategy": "web_search_fallback",
            "node_latencies": {"research_node": node_latency_ms},
            "agent_log": [
                f"[ResearchAgent] Query: '{query}' | "
                f"No structured RAG output found — web_search fallback used"
                f"Latency: {node_latency_ms} ms"
            ],
            
        }

    # Map rag_tool.py fields to MultiAgentState fields
    research_context = tool_output.get("context", "")
    sources = tool_output.get("sources", [])
    confidence = tool_output.get("confidence", "low")
    action = tool_output.get("action", "retry_or_abstain")
    answerable = tool_output.get("answerable", False)
    retrieval_strategy = tool_output.get("retrieval_strategy", "unknown")

    log_entry = (
        f"[ResearchAgent] Query: '{query}' | "
        f"Confidence: {confidence} | "
        f"Action: {action} | "
        f"Answerable: {answerable} | "
        f"Strategy: {retrieval_strategy} | "
        f"Sources: {len(sources)}"
        f"Latency: {node_latency_ms} ms"
    )

    return {
        "research_context": research_context,
        "sources": sources,
        "confidence": confidence,
        "action": action,
        "answerable": answerable,
        "retrieval_strategy": retrieval_strategy,
        "node_latencies": {"research_node": node_latency_ms},
        "agent_log": [log_entry]
    }

if __name__ == "__main__":
    # Test the research_node function
    test_state = cast(MultiAgentState, {
        "query": "What is ReAct prompting framework for LLM agents?"
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
    print(f"Agent Log: {result['agent_log']}")
