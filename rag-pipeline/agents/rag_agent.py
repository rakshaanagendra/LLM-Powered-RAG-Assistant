from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq # for deployment as Ollama does not support fre cloud servers as RAM is low
from groq import BadRequestError
import sys
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------------------------------------------
# Path setup — same pattern as rag_tool.py
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.rag_tool import rag_search
from agents.tavily_tool import web_search

# -----------------------------------------------------------------------
# State
# Why: LangGraph needs a state object to pass between nodes
# add_messages is a special reducer — it appends new messages
# to the list rather than overwriting it
# -----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------------------------------------------------
# LLM setup
# Why: we bind the tool to the LLM so it knows the tool exists
# and can choose to call it
# -----------------------------------------------------------------------
tools = [rag_search, web_search]

# llm = ChatOllama(model="qwen2.5:7b")
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

SYSTEM_PROMPT = """You are a research assistant with access to two tools:

1. rag_search — use this for questions about AI, LLMs, RAG pipelines,
   retrieval systems, agents, multi-agent systems, LLM engineering,
   fine-tuning, and related research topics from academic papers.

2. web_search — use this for questions where the answer could be
   current, recent, or subject to change: current events, live prices,
   sports results, current officeholders, weather, or any topic where
   being out-of-date would make the answer wrong.

3. If the question is basic math, unit conversion, spelling, word
   definitions/opposites, casual conversation, or a stable fact that
   doesn't change over time (e.g. historical dates, scientific constants) —
   answer it directly yourself, without calling any tool.

4. EXCEPTION to rule 3: if the question asks about a specific research
   paper, its authors, findings, or claims — always use rag_search, even
   if you believe you already know the answer. This ensures the answer
   is grounded in the actual paper and properly cited.

   RULES:
1. Always pass the user's COMPLETE original question to the tool. Never shorten it.
2. Pick the right tool based on the query topic.
3. After getting the tool result, check the 'action' field in rag_search results:
   - 'generate' → answer confidently
   - 'generate_cautiously' → answer but mention uncertainty
   - 'retry_or_abstain' → tell the user you cannot find relevant information
4. Always cite sources in your answer.
"""

MAX_TOOL_CALL_RETRIES = 2

def llm_node(state: AgentState) -> AgentState:
    """LLM reasons and decides whether to call a tool or answer directly."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    for _ in range(MAX_TOOL_CALL_RETRIES):
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except BadRequestError as e:
            body = e.body if isinstance(e.body, dict) else {}
            if body.get("error", {}).get("code") != "tool_use_failed":
                raise
            # Groq/Llama occasionally emits a malformed tool-call instead of
            # a structured one — a stochastic decoding glitch, not a content
            # failure, so retrying the same messages usually succeeds.

    # Retries exhausted — fall back to a direct answer with no tool access
    # rather than crashing the graph.
    response = llm.invoke(messages)
    return {"messages": [response]}
# ToolNode handles tool execution automatically
# It reads the tool call from the last LLM message,
# executes the right tool, and returns the result as a message
tool_node = ToolNode(tools)

# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm_node", llm_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm_node")

# tools_condition returns "tools" or END — the node must be named "tools"
graph_builder.add_conditional_edges(
    "llm_node",
    tools_condition,
)

graph_builder.add_edge("tools", "llm_node")

checkpointer = MemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
if __name__ == "__main__":

    # print("=== TEST 1: AI TOPIC → should use rag_search ===")
    # result = graph.invoke(
    #     {"messages": [{"role": "user", "content": "What is ReAct prompting framework for LLM agents?"}]},
    #     config={"recursion_limit": 10}
    # )
    # print(result["messages"][-1].content)

    # print("\n=== TEST 2: GENERAL KNOWLEDGE → should use web_search ===")
    # result2 = graph.invoke(
    #     {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
    #     config={"recursion_limit": 10}
    # )
    # print(result2["messages"][-1].content)

    # print("\n=== TEST 3: EDGE CASE → who invented transformers? ===")
    # result3 = graph.invoke(
    #     {"messages": [{"role": "user", "content": "Who invented the transformer architecture?"}]},
    #     config={"recursion_limit": 10}
    # )
    # print(result3["messages"][-1].content)

    # ------------------------------------------------------------------
    # TEST 1: Multi-turn memory — same thread_id across 3 turns
    # Expected: turn 2 and 3 correctly resolve "it" and "that comparison"
    # because the agent sees the full message history from this thread
    # ------------------------------------------------------------------
    print("=== TEST 1: MULTI-TURN MEMORY (same thread) ===")

    thread_a: RunnableConfig = {
        "configurable": {"thread_id": "session_memory_test"},
        "recursion_limit": 10
    }

    turn1 = graph.invoke(
        {"messages": [{"role": "user", "content": "What is the ReAct prompting framework for LLM agents?"}]},
        config=thread_a
    )
    print("Turn 1:", turn1["messages"][-1].content)

    turn2 = graph.invoke(
        {"messages": [{"role": "user", "content": "Can you give me a concrete example of it?"}]},
        config=thread_a  # same thread_id — agent sees full history, knows "it" = ReAct
    )
    print("\nTurn 2 (refers to 'it' from turn 1):", turn2["messages"][-1].content)

    turn3 = graph.invoke(
        {"messages": [{"role": "user", "content": "How does that compare to chain-of-thought prompting?"}]},
        config=thread_a  # agent still has full history from turns 1 and 2
    )
    print("\nTurn 3 (compares to prior context):", turn3["messages"][-1].content)

    # ------------------------------------------------------------------
    # TEST 2: Memory isolation — different thread_id starts completely fresh
    # Expected: agent has NO knowledge of the ReAct conversation above
    # ------------------------------------------------------------------
    print("\n=== TEST 2: MEMORY ISOLATION (different thread) ===")

    thread_b: RunnableConfig = {
        "configurable": {"thread_id": "session_isolation_test"},
        "recursion_limit": 10
    }

    isolation_test = graph.invoke(
        {"messages": [{"role": "user", "content": "What did we just discuss?"}]},
        config=thread_b  # fresh thread — no history at all
    )
    print("Fresh thread response (should NOT know about ReAct):", isolation_test["messages"][-1].content)