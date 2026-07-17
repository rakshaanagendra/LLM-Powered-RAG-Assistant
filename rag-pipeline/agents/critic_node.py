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
CRITIC_SYSTEM_PROMPT = """You are a grounding critic. Your job is to catch fabricated or 
misattributed claims — NOT to demand exact wording matches.

Your job is to check whether a given answer is grounded in the provided research context.

Rules:
1. Read the query, the final answer, and the research context carefully
2. Check every factual claim in the final answer against the research context
3. A claim is GROUNDED if the underlying fact appears in the context — even if:
   - it's paraphrased or reworded differently than the context
   - it's assembled by combining facts from two or more separate chunks
   - it's a reasonable, direct restatement of what the context says
4. A claim is UNGROUNDED only if:
   - the fact does not appear anywhere in the context (fabrication), OR
   - the claim attributes information to the wrong entity/topic (e.g. describing 
     a different "Phoenix" than the one the context is actually about), OR
   - the claim states a specific detail (a number, name, date, or mechanism) that 
     contradicts or is absent from the context
4b. Before treating any chunk as support for a claim about a specific named entity 
    (a tool, model, method, framework, or product), verify that the chunk actually 
    names or clearly refers to that same entity. A chunk discussing the same general 
    subject area, without naming the specific entity the claim is about, does NOT 
    count as support — this is misattribution, not synthesis. If two different things 
    happen to share a name, check what the chunk's actual subject is, not just 
    whether a label or keyword matches.
5. Do NOT penalize a claim just because it uses different words than the context, 
   or because it connects information from multiple chunks — that is normal synthesis, 
   not a grounding failure
6. Set answer_grounded to true only if every claim passes the GROUNDED test in rule 3/4. 
   If even one claim is fabricated or misattributed, set answer_grounded to false
7. Write a critique explaining your decision — be specific about which claims are 
   grounded or not, and WHY (fabricated vs. misattributed vs. genuinely unsupported)

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
    source_type = state.get("source_type", "rag")

    # NEW hard gate — must come FIRST. When source_type is
    # "general_knowledge", there is no research_context to check
    # grounding against — the question "is this grounded in the
    # context" is meaningless here, not just hard. Rather than asking
    # the LLM to fake-check something that doesn't apply (which is what
    # produced the false "grounded in research context" verdict on the
    # cake/photosynthesis queries), we label it honestly instead.
    if source_type == "general_knowledge":
        node_latency_ms = round((time.time() - node_start) * 1000, 2)
        return {
            "answer_grounded": True,   # not rejected — just not a grounding
                                        # question in the first place
            "critique": (
                "Not applicable — no retrieval was performed. This answer "
                "was sourced directly from the model's own general "
                "knowledge, not verified against any retrieved context."
            ),
            "node_latencies": {"critic_node": node_latency_ms},
            "agent_log": [
                f"[CriticAgent] Skipped grounding check — source_type: "
                f"general_knowledge | Latency: {node_latency_ms}ms"
            ],
        }

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
        "abstained": False,
        "source_type": "rag",
        "node_latencies": {},
        "agent_log": []
    })

    result = critic_node(test_state)
    print("=== Critic Node Test Result ===")
    print("Answer Grounded:", result["answer_grounded"])
    print("Critique:", result["critique"])
    print("Node Latencies:", result["node_latencies"])
    print("Agent Log:", result["agent_log"])