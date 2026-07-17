"""
generate_critic_eval_inputs.py

Runs research_node -> writer_node on a fixed set of queries and saves
the resulting (query, context, answer) triples to a JSON file.

This is NOT the full graph. No critic, no retry loop. Just the two
nodes chained manually, so we can capture real writer output to later
feed into critic_node in isolation.

Run this from wherever research_node.py and writer_node.py already
import correctly in your repo (same place you run rag_agent.py from).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add rag-pipeline/ to the search path so "agents.*" can be imported,
# regardless of which folder you run this script from.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from groq import BadRequestError
from agents.research_node import research_node
from agents.writer_node import writer_node
from agents.state import MultiAgentState

MAX_ATTEMPTS = 3


def research_node_with_retry(state: MultiAgentState) -> dict:
    """
    Wraps research_node with retry-on-BadRequestError, same pattern as
    eval_runner_tool_selection.py. The underlying issue: llama-3.3-70b-versatile
    on Groq intermittently emits malformed function-call syntax
    (e.g. '<function=rag_search {...}></function>' as plain text) instead of
    a real structured tool call. Groq rejects this with tool_use_failed.
    It's a sampling issue, not a logic bug -- retrying the same input often
    succeeds on the 2nd attempt.
    """
    last_error: BadRequestError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return research_node(state)
        except BadRequestError as e:
            last_error = e
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: tool_use_failed. Retrying...")
            time.sleep(2)
    assert last_error is not None
    raise last_error

# -----------------------------------------------------------------------
# Query set — mix of known-bad, untested, ambiguous, and OOD queries
# -----------------------------------------------------------------------
QUERIES = [
    # Known-bad (from retrieval eval bug list)
    {"query_id": "q001", "query": "What is the ReAct framework?", "tag": "known_bad_reranker"},
    {"query_id": "q008", "query": "What is Retrieval-Augmented Generation?", "tag": "known_bad_sort_bug"},
    {"query_id": "q029", "query": "How does Phoenix support LLM evaluation and observability?", "tag": "known_bad_corpus_gap"},
    {"query_id": "q048", "query": "Describe how persistence works across LangGraph runs.", "tag": "known_bad_unknown"},

    # Untested, spread across corpus folders
    {"query_id": "q003", "query": "Compare ReAct and Reflexion approaches to agent reasoning.", "tag": "untested"},
    {"query_id": "q013", "query": "Why does the 'lost in the middle' problem occur in long-context LLMs?", "tag": "untested"},
    {"query_id": "q022", "query": "Compare AutoGen and CAMEL approaches to multi-agent orchestration.", "tag": "untested"},
    {"query_id": "q025", "query": "What is LoRA and how does it enable parameter-efficient fine-tuning?", "tag": "untested"},
    {"query_id": "q028", "query": "What is RAGAS and what does it measure?", "tag": "untested"},
    {"query_id": "q035", "query": "How does speculative decoding speed up LLM inference?", "tag": "untested"},
    {"query_id": "q046", "query": "What is LangGraph and what problem does it solve?", "tag": "untested"},

    # Known ambiguous / domain-gate edge case
    {"query_id": "q017", "query": "Who proposed the transformer architecture in 'Attention Is All You Need'?", "tag": "ambiguous_domain_gate"},

    # OOD — domain gate has 0/8 detection rate, high risk of false answerable=True
    {"query_id": "q037", "query": "What is the capital of France?", "tag": "ood"},
    {"query_id": "q043", "query": "What is the plot of the movie Inception?", "tag": "ood"},
]


def run_pipeline_for_query(query_id: str, query: str, tag: str) -> dict:
    """Chains research_node -> writer_node for one query, returns the full record."""
    state: MultiAgentState = {
        "query": query,
        "research_context": "",
        "sources": [],
        "confidence": "",
        "action": "",
        "answerable": False,
        "retrieval_strategy": "",
        "critique": "",
        "final_answer": "",
        "answer_grounded": False,
        "retry_count": 0,
        "agent_log": [],
        "node_latencies": {},
    }

    research_start = time.time()
    research_output = research_node_with_retry(state)
    state.update(research_output)
    research_time = round(time.time() - research_start, 2)

    writer_start = time.time()
    writer_output = writer_node(state)
    state.update(writer_output)
    writer_time = round(time.time() - writer_start, 2)

    return {
        "query_id": query_id,
        "query": query,
        "tag": tag,
        "research_context": state.get("research_context", ""),
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", ""),
        "action": state.get("action", ""),
        "answerable": state.get("answerable", None),
        "retrieval_strategy": state.get("retrieval_strategy", ""),
        "final_answer": state.get("final_answer", ""),
        "research_time_sec": research_time,
        "writer_time_sec": writer_time,
        # You fill these in after reading the output:
        "human_label": None,          # "good" or "bad"
        "human_label_reason": ""
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated query_ids to run (e.g. q022,q043). Defaults to all queries.",
    )
    args = parser.parse_args()

    queries = QUERIES
    if args.query_ids:
        wanted = {qid.strip() for qid in args.query_ids.split(",")}
        queries = [item for item in QUERIES if item["query_id"] in wanted]
        missing = wanted - {item["query_id"] for item in queries}
        if missing:
            raise ValueError(f"Unknown query_id(s): {', '.join(sorted(missing))}")

    results = []
    for item in queries:
        print(f"Running {item['query_id']}: {item['query']}")
        try:
            record = run_pipeline_for_query(item["query_id"], item["query"], item["tag"])
        except Exception as e:
            print(f"  FAILED: {e}")
            record = {
                "query_id": item["query_id"],
                "query": item["query"],
                "tag": item["tag"],
                "error": str(e),
                "human_label": None,
                "human_label_reason": ""
            }
        results.append(record)
        time.sleep(2)  # Groq free-tier rate limit spacing, same as your other eval runners

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"critic_eval_inputs_{timestamp}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} records to {out_path.resolve()}")


if __name__ == "__main__":
    main()