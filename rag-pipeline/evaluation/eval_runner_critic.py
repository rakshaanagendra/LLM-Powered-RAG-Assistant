"""
eval_runner_critic.py

Runs critic_node in isolation against human-labeled (query, context, answer)
triples from critic_eval_inputs.json, and checks whether critic_node's
answer_grounded verdict matches the human label.

Distinguishes two failure types:
- false_negative: critic said grounded, human said bad  -> DANGEROUS (hallucination ships)
- false_positive: critic said ungrounded, human said good -> wasted retry, not dangerous

"""
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

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
from agents.critic_node import critic_node
from agents.state import MultiAgentState

candidates = sorted(Path(".").glob("critic_eval_inputs_*.json"))
if not candidates:
    raise FileNotFoundError("No critic_eval_inputs_*.json file found in this directory.")
INPUT_PATH = candidates[-1]
print(f"Using input file: {INPUT_PATH}")

MAX_ATTEMPTS = 2


def label_to_expected_grounded(human_label: str) -> bool:
    """'good' -> critic should say grounded=True. 'bad' -> critic should say grounded=False."""
    return human_label == "good"


def critic_node_with_retry(state: MultiAgentState) -> dict:
    """
    Same retry pattern as research_node_with_retry in generate_critic_eval_inputs.py.
    critic_node runs on the same Groq model (llama-3.3-70b-versatile) that
    intermittently emits malformed tool-call syntax. Retrying the same input
    once is usually enough for it to succeed on the 2nd attempt.
    """
    last_error: BadRequestError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return critic_node(state)
        except BadRequestError as e:
            last_error = e
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: tool_use_failed. Retrying...")
            time.sleep(2)
    assert last_error is not None
    raise last_error


def run_one(record: dict) -> dict:
    state: MultiAgentState = {
        "query": record["query"],
        "research_context": record.get("research_context", ""),
        "sources": record.get("sources", []),
        "confidence": record.get("confidence", "medium"),
        "action": record.get("action", "generate"),
        "answerable": record.get("answerable", True),
        "retrieval_strategy": record.get("retrieval_strategy", "dense"),
        "critique": "",
        "final_answer": record.get("final_answer", ""),
        "answer_grounded": False,
        "retry_count": 0,
        "abstained": False,
        "agent_log": [],
        "node_latencies": {},
    }

    result = critic_node_with_retry(state)
    predicted_grounded = result["answer_grounded"]
    expected_grounded = label_to_expected_grounded(record["human_label"])

    match = predicted_grounded == expected_grounded

    if match:
        failure_type = None
    elif predicted_grounded and not expected_grounded:
        failure_type = "false_negative"   # critic missed a bad answer -- dangerous
    else:
        failure_type = "false_positive"   # critic over-flagged a good answer

    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "tag": record.get("tag", ""),
        "human_label": record["human_label"],
        "human_label_reason": record.get("human_label_reason", ""),
        "expected_grounded": expected_grounded,
        "predicted_grounded": predicted_grounded,
        "match": match,
        "failure_type": failure_type,
        "critic_critique": result["critique"],
    }


def main():
    records = json.loads(INPUT_PATH.read_text())

    labeled = [r for r in records if r.get("human_label") in ("good", "bad")]
    skipped = [r for r in records if r.get("human_label") not in ("good", "bad")]

    if skipped:
        print(f"Skipping {len(skipped)} unlabeled record(s): "
              f"{[r['query_id'] for r in skipped]}")

    results = []
    for record in labeled:
        print(f"Evaluating {record['query_id']}: {record['query']}")
        try:
            results.append(run_one(record))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "query_id": record["query_id"],
                "query": record["query"],
                "error": str(e),
                "match": None,
                "failure_type": "runner_error",
            })
        time.sleep(2)  # Groq free-tier rate limit spacing

    total = len(results)
    matched = sum(1 for r in results if r["match"] is True)
    false_negatives = [r for r in results if r["failure_type"] == "false_negative"]
    false_positives = [r for r in results if r["failure_type"] == "false_positive"]

    summary = {
        "total_evaluated": total,
        "matched": matched,
        "accuracy": round(matched / total, 3) if total else None,
        "false_negative_count": len(false_negatives),
        "false_negative_query_ids": [r["query_id"] for r in false_negatives],
        "false_positive_count": len(false_positives),
        "false_positive_query_ids": [r["query_id"] for r in false_positives],
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"critic_eval_results_{timestamp}.json")

    output = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(output, indent=2))

    print("\n=== Critic Eval Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()