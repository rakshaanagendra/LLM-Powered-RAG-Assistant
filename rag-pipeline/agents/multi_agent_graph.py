import sys
from pathlib import Path
from typing import cast
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.state import MultiAgentState
from agents.research_node import research_node
from agents.writer_node import writer_node
from agents.critic_node import critic_node

MAX_RETRIES = 1

def route_after_critic(state: MultiAgentState) -> str:
    if state.get("answer_grounded"):
        return "end"
    if state.get("retry_count", 0) > MAX_RETRIES:
        return "end"
    return "retry"

# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------
graph_builder = StateGraph(MultiAgentState)

# Add nodes
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("writer_node", writer_node)
graph_builder.add_node("critic_node", critic_node)

# Wire edges — straight pipeline, no loops
graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "writer_node")
graph_builder.add_edge("writer_node", "critic_node")
graph_builder.add_conditional_edges(
    "critic_node",
    route_after_critic,
    {"retry": "writer_node", "end": END}
)
# Compile — no checkpointer needed, outer graph is stateless
# Memory is handled at the research_node level via uuid thread_id
graph = graph_builder.compile()

# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke(cast(MultiAgentState, 
        {"query": "What is ReAct prompting framework for LLM agents?"}
    ))

    print("=== Multi-Agent Pipeline Result ===")
    print("\nQuery:", result["query"])
    print("\nResearch Context (first 300 chars):", result["research_context"][:300])
    print("\nSources:", result["sources"])
    print("\nConfidence:", result["confidence"])
    print("\nAction:", result["action"])
    print("\nFinal Answer:", result["final_answer"])
    print("\nAnswer Grounded:", result["answer_grounded"])
    print("\n Retry Count:",result["retry_count"])
    print("\nCritique:", result["critique"])
    print("\nAgent Log:")
    for entry in result["agent_log"]:
        print(" -", entry)