"""
eval_runner_loop.py

Pillar 2 — Loop efficiency eval.

Runs the FULL multi-agent graph (research_node -> writer_node -> critic_node
-> [retry or abstain]) end-to-end for each query in loop_eval_queries.json,
and checks whether the loop's structural behavior (retried? abstained?
grounded?) matches what we expected for that query's category.

IMPORTANT LIMITATION (documented, not hidden):
This script can only check STRUCTURE (did it retry, did it abstain, did it
end grounded) — it cannot check TRUTH (was the critic actually right to
approve the answer). A query can pass the structural check and still contain
a false-negative critic verdict (see the live ReAct test — critic said
grounded=True while citing a chunk that didn't support the claim). Any
known_bad or retry_candidate query that ends grounded=True is flagged for
manual review instead of being silently marked "pass".
"""
from dotenv import load_dotenv
load_dotenv()

import json
import sys
import time
from pathlib import Path
from typing import cast

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from groq import BadRequestError
from agents.multi_agent_graph import graph
from agents.state import MultiAgentState

QUERIES_PATH = Path("loop_eval_queries.json")
MAX_ATTEMPTS = 2  # retry-once wrapper for Groq's intermittent BadRequestError

# Expected structural outcome per tag — used only to flag mismatches for review,
# not as an automatic pass/fail on truth.
# NOTE: "ood" no longer means "must abstain". Since general_knowledge and
# web_search paths were fixed to handle these correctly, the real failure
# mode to catch now is DOMAIN GATE LEAK — an OOD query ending up with
# source_type "rag" would mean the gate let something through it shouldn't
# have. general_knowledge or web_search are both correct; rag is not.
EXPECTED_OUTCOME = {
    "known_bad": "abstained",
    "should_pass_first_try": "grounded_first_try",
    "ood": "not_rag",   # checked via source_type, not outcome bucket
    "retry_candidate": "grounded_or_abstained",  # behavior tells us which; both are fine
}


def invoke_with_retry(query: str) -> dict:
    """Same retry pattern as eval_runner_critic.py — Groq's llama-3.3-70b-versatile
    intermittently emits malformed tool-call syntax. Retry once."""
    last_error: BadRequestError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return graph.invoke(cast(MultiAgentState, {"query": query}))
        except BadRequestError as e:
            last_error = e
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: tool_use_failed. Retrying...")
            time.sleep(2)
    assert last_error is not None
    raise last_error


def classify_outcome(result: dict) -> str:
    """Turn the raw result state into one of 4 structural buckets."""
    abstained = result.get("abstained", False)
    grounded = result.get("answer_grounded", False)
    retry_count = result.get("retry_count", 0)

    if abstained:
        return "abstained"
    if grounded and retry_count == 0:
        return "grounded_first_try"
    if grounded and retry_count > 0:
        return "grounded_after_retry"
    return "unexpected"  # grounded=False, abstained=False -- should not be possible
                          # if abstain_node is wired correctly; a hit here means
                          # the routing itself is broken, not just a bad answer


def check_structural_match(tag: str, outcome: str, source_type: str) -> bool:
    expected = EXPECTED_OUTCOME.get(tag, "")
    if expected == "abstained":
        return outcome == "abstained"
    if expected == "grounded_first_try":
        return outcome == "grounded_first_try"
    if expected == "not_rag":
        # The only failure that matters here: domain gate leaked an
        # OOD query into the RAG path. general_knowledge or web_search
        # are both fine — either means the gate correctly kept it out
        # of your corpus retrieval.
        return source_type in ("general_knowledge", "web_search")
    if expected == "grounded_or_abstained":
        return outcome in ("abstained", "grounded_after_retry", "grounded_first_try")
    return False


def run_one(record: dict) -> dict:
    query_id = record["query_id"]
    query = record["query"]
    tag = record["tag"]

    print(f"Running {query_id} [{tag}]: {query}")
    result = invoke_with_retry(query)

    outcome = classify_outcome(result)
    source_type = result.get("source_type", "unknown")
    structural_match = check_structural_match(tag, outcome, source_type)

    # Flag for manual review: known_bad / retry_candidate queries that came
    # back grounded=True need a human to actually read the critique, since
    # the critic itself has a documented false-negative rate (Pillar 3).
    # For "ood" queries, flag for review if source_type ended up "rag" —
    # that's the domain gate leak case, worth reading the full trace.
    needs_manual_review = (
        (tag in ("known_bad", "retry_candidate")
         and outcome in ("grounded_first_try", "grounded_after_retry"))
        or (tag == "ood" and source_type == "rag")
    )

    return {
        "query_id": query_id,
        "query": query,
        "tag": tag,
        "outcome": outcome,
        "source_type": source_type,
        "expected_outcome": EXPECTED_OUTCOME.get(tag, ""),
        "structural_match": structural_match,
        "needs_manual_review": needs_manual_review,
        "retry_count": result.get("retry_count", 0),
        "answer_grounded": result.get("answer_grounded", False),
        "abstained": result.get("abstained", False),
        "final_answer": result.get("final_answer", ""),
        "critique": result.get("critique", ""),
        "node_latencies": result.get("node_latencies", {}),
        "sources": result.get("sources", []),
    }


def main():
    records = json.loads(QUERIES_PATH.read_text())

    results = []
    for record in records:
        try:
            results.append(run_one(record))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "query_id": record["query_id"],
                "query": record["query"],
                "tag": record["tag"],
                "error": str(e),
                "outcome": "runner_error",
                "structural_match": None,
                "needs_manual_review": True,
            })
        time.sleep(2)  # Groq free-tier rate limit spacing

    total = len(results)
    matched = sum(1 for r in results if r.get("structural_match") is True)
    needs_review = [r["query_id"] for r in results if r.get("needs_manual_review")]
    unexpected = [r["query_id"] for r in results if r.get("outcome") == "unexpected"]

    # Retry stats
    retry_counts = [r.get("retry_count", 0) for r in results if "retry_count" in r]
    zero_retry = sum(1 for c in retry_counts if c == 0)
    one_retry = sum(1 for c in retry_counts if c == 1)
    hit_max = sum(1 for c in retry_counts if c > 1)

    summary = {
        "total_evaluated": total,
        "structural_matches": matched,
        "structural_match_rate": round(matched / total, 3) if total else None,
        "needs_manual_review_count": len(needs_review),
        "needs_manual_review_query_ids": needs_review,
        "unexpected_outcome_count": len(unexpected),
        "unexpected_outcome_query_ids": unexpected,
        "retry_distribution": {
            "0_retries": zero_retry,
            "1_retry": one_retry,
            "hit_max_retries": hit_max,
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"loop_eval_results_{timestamp}.json")
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n=== Loop Efficiency Eval Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()