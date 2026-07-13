import json
import re
import sys
import time
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.state import MultiAgentState

# -----------------------------------------------------------------------
# LLM — llama-3.3-70b-versatile for strict grounding evaluation
# -----------------------------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

# -----------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """You are a strict answer grounding critic.

Your job is to check whether a given answer is grounded in the provided research context.

Rules:
1. Read the query, the final answer, and the research context carefully
2. Check every claim in the final answer against the research context
3. If ALL claims are directly supported by the context, set answer_grounded to true
4. If ANY claim cannot be traced back to the context — even minor connecting language or assumptions — set answer_grounded to false
5. Write a critique explaining your decision — be specific about which claims are grounded or not
6. Do NOT infer or assume. If the exact claim is not present verbatim or near-verbatim in the context, mark it as ungrounded.

Respond ONLY with this exact JSON format, no other text, no markdown:
{"answer_grounded": true or false, "critique": "your explanation here"}"""

# -----------------------------------------------------------------------
# Critic Node
# -----------------------------------------------------------------------
def critic_node(state: MultiAgentState) -> dict:
    """
    Validates whether final_answer is grounded in research_context.
    Hard gate on empty final_answer.
    LLM call compares answer against context for grounding.
    Records its own latency into node_latencies.
    """
    # ADDED: start timer before any work begins
    node_start = time.time()

    query = state["query"]
    final_answer = state.get("final_answer", "")
    research_context = state.get("research_context", "")

    # Hard gate — nothing to critique if writer produced no answer
    if not final_answer:
        node_latency_ms = round((time.time() - node_start) * 1000, 2)
        return {
            "answer_grounded": False,
            "critique": "No answer was generated to critique.",
            "node_latencies": {"critic_node": node_latency_ms},
            "agent_log": [f"[CriticAgent] Skipped — final_answer is empty | Latency: {node_latency_ms}ms"]
        }

    human_message = f"""Query: {query}

Final Answer to critique:
{final_answer}

Research Context to compare against:
{research_context}

Now evaluate whether the final answer is grounded in the research context."""

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=human_message)
    ]

    response = llm.invoke(messages)
    raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

    clean = raw.replace("```json", "").replace("```", "").strip()
    clean = re.sub(r'[\n\r\t]', ' ', clean)

    try:
        parsed = json.loads(clean)
        answer_grounded = bool(parsed.get("answer_grounded", False))
        critique = parsed.get("critique", "")
    except (json.JSONDecodeError, KeyError) as e:
        answer_grounded = False
        critique = f"Critic parse failed — raw output: {raw}"

    # ADDED: stop timer after LLM call completes
    node_latency_ms = round((time.time() - node_start) * 1000, 2)

    log_entry = (
        f"[CriticAgent] Query: '{query}' | "
        f"Grounded: {answer_grounded} | "
        f"Critique length: {len(critique)} chars | "
        f"Latency: {node_latency_ms}ms"
    )

    retry_count=state.get("retry_count",0)
    if not answer_grounded:
        retry_count+=1

    return {
        "answer_grounded": answer_grounded,
        "critique": critique,
        "retry_count":retry_count,
        "node_latencies": {"critic_node": node_latency_ms},
        "agent_log": [log_entry]
    }


# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_state = MultiAgentState({
        "query": "What is ReAct prompting framework for LLM agents?",
        "final_answer": (
            "ReAct is a prompting framework that enhances LLM agents by interleaving "
            "reasoning and acting. It allows agents to take actions, observe results, "
            "and incorporate observations into future reasoning steps."
        ),
        "research_context": (
            "ReAct interleaves reasoning and acting to create synergy between the two. "
            "It was shown to improve performance on language and decision-making tasks. "
            "The framework generates reasoning traces and action plans simultaneously."
        ),
        "sources": ["REACT.pdf"],
        "confidence": "medium",
        "action": "generate_cautiously",
        "answerable": True,
        "retrieval_strategy": "dense",
        "critique": "",
        "answer_grounded": False,
        "retry_count": 0,
        "node_latencies": {},
        "agent_log": []
    })

    result = critic_node(test_state)
    print("=== Critic Node Test Result ===")
    print("Answer Grounded:", result["answer_grounded"])
    print("Critique:", result["critique"])
    print("Node Latencies:", result["node_latencies"])
    print("Agent Log:", result["agent_log"])