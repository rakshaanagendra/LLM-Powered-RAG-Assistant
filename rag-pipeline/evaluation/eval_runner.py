"""
Retrieval evaluation runner.

Loads evaluation/eval_queries.json, runs every query through
MultiQueryHybridRetriever, scores the results using metrics.py,
and writes:
  - evaluation/eval_results.json   (per-query detailed results)
  - evaluation/eval_summary.json   (aggregated pass/fail report)

Run from anywhere:  python eval_runner.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# --- Path setup so this script can import from retrieval/ as a sibling folder ---
# evaluation/eval_runner.py now lives INSIDE rag-pipeline/, as a peer of
# retrieval/, generation/, observability/, caching/, agents/:
#   rag-pipeline/evaluation/eval_runner.py
#   rag-pipeline/retrieval/multi_query_hybrid.py
# So we only need to go up ONE level (not two) to reach rag-pipeline/ itself.
RAG_PIPELINE_ROOT = Path(__file__).resolve().parent.parent

if str(RAG_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_PIPELINE_ROOT))

from retrieval.multi_query_hybrid import MultiQueryHybridRetriever  # noqa: E402
import metrics  # noqa: E402  (metrics.py lives in this same evaluation/ folder)


EVAL_QUERIES_PATH = Path(__file__).resolve().parent / "eval_queries.json"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"
EVAL_SUMMARY_PATH = Path(__file__).resolve().parent / "eval_summary.json"

TOP_K_FOR_RANKING = 5  # how many reranked chunks we treat as "retrieved" for scoring


def build_source_key(chunk: dict) -> str:
    """
    Reconstruct 'category/title' (e.g. 'agents/REACT') from a chunk dict,
    matching the format used in expected_sources within eval_queries.json.

    Confirmed from loader.py / chunker.py:
      - chunk['category'] = folder name  (e.g. 'agents')
      - chunk['title']    = filename without extension (e.g. 'REACT')
      - chunk['source']   = filename WITH extension (e.g. 'REACT.pdf') -- fallback only
    """
    category = chunk.get("category") or "unknown"
    title = chunk.get("title") or os.path.splitext(chunk.get("source", "unknown"))[0]
    return f"{category}/{title}"


def run_single_query(retriever: MultiQueryHybridRetriever, query_entry: dict, debug: bool = False) -> dict:
    """
    Runs one query through the retriever, scores it, and returns
    the query_entry dict with all result fields populated.
    """
    query = query_entry["query"]
    expected_sources = query_entry["expected_sources"]

    result = query_entry.copy()

    try:
        pipeline_result = retriever.adaptive_search_with_retry(
            query,
            num_queries=4,
            retrieve_k=40,
            final_k=10,
            context_top_k=5,
        )
    except Exception as exc:
        # A single query failing must NOT kill the whole eval run.
        # Record the failure and move on -- this is the "aggressive
        # surfacing of silent failures" principle from your own notes.
        result["error"] = f"Pipeline raised an exception: {exc}"
        result["retrieved_sources"] = []
        result["mrr"] = 0.0
        result["recall_at_5"] = 0.0
        result["precision_at_5"] = 0.0
        result["hit_at_3"] = False
        result["hit_at_5"] = False
        result["confidence"] = None
        result["action"] = None
        result["is_ood_predicted"] = None
        result["can_answer"] = None
        result["combined_fit"] = None
        return result

    final_result = pipeline_result["final_result"]
    reranked_chunks = final_result["reranked_chunks"]
    diagnostics = final_result["diagnostics"]
    confidence_route = pipeline_result.get("confidence_route", {})

    if debug and reranked_chunks:
        print("\n    [DEBUG] Raw first reranked chunk fields:")
        first_chunk = reranked_chunks[0]
        for key in ("category", "title", "source", "source_path"):
            print(f"    [DEBUG]   {key} = {first_chunk.get(key, '<<MISSING>>')!r}")
        print(f"    [DEBUG]   build_source_key() -> {build_source_key(first_chunk)!r}\n")

    retrieved_sources = [build_source_key(chunk) for chunk in reranked_chunks[:TOP_K_FOR_RANKING]]

    result["retrieved_sources"] = retrieved_sources
    result["mrr"] = round(metrics.mean_reciprocal_rank(retrieved_sources, expected_sources), 4)
    result["recall_at_5"] = round(metrics.recall_at_k(retrieved_sources, expected_sources, k=5), 4)
    result["precision_at_5"] = round(metrics.precision_at_k(retrieved_sources, expected_sources, k=5), 4)
    result["hit_at_3"] = metrics.hit_at_k(retrieved_sources, expected_sources, k=3)
    result["hit_at_5"] = metrics.hit_at_k(retrieved_sources, expected_sources, k=5)

    result["confidence"] = confidence_route.get("confidence")
    result["action"] = confidence_route.get("action")
    result["is_ood_predicted"] = diagnostics.get("domain_gate", {}).get("is_ood")
    result["can_answer"] = diagnostics.get("answerability", {}).get("can_answer")
    result["combined_fit"] = diagnostics.get("domain_gate", {}).get("combined_fit")

    return result


def summarize(results: list) -> dict:
    """
    Aggregate per-query results into mean scores, split by
    in-domain vs OOD, since averaging them together would be
    misleading (OOD queries always have mrr=0 by design).
    """
    in_domain = [r for r in results 
                 if not r["is_ood_ground_truth"]
                 and
                 not r.get("corpus_gap", False)]  # corpus_gap queries get their own section below, excluded from both in-domain and OOD scoring
    ood = [r for r in results if r["is_ood_ground_truth"]]

    corpus_gap_queries = [r for r in results if r.get("corpus_gap", False)]
    
    def avg(items, field):
        vals = [i[field] for i in items if i.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def rate(items, field):
        vals = [i[field] for i in items if i.get(field) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None

    # OOD detection accuracy: did is_ood_predicted match is_ood_ground_truth?
    ood_detection_correct = sum(
        1 for r in results
        if r.get("is_ood_predicted") == r["is_ood_ground_truth"]
    )
    ood_detection_accuracy = round(ood_detection_correct / len(results), 4) if results else None

    errors = [r["query_id"] for r in results if r.get("error")]

    return {
        "total_queries": len(results),
        "in_domain_queries": len(in_domain),
        "ood_queries": len(ood),
        "queries_with_errors": errors,

        "in_domain_metrics": {
            "mean_mrr": avg(in_domain, "mrr"),
            "mean_recall_at_5": avg(in_domain, "recall_at_5"),
            "mean_precision_at_5": avg(in_domain, "precision_at_5"),
            "hit_rate_at_3": rate(in_domain, "hit_at_3"),
            "hit_rate_at_5": rate(in_domain, "hit_at_5"),
        },

        "corpus_gap_metrics": {
            "count": len(corpus_gap_queries),
            "correctly_abstained": sum(1 for r in corpus_gap_queries if not r.get("can_answer", True)),
        },

        "domain_gate_metrics": {
            "ood_detection_accuracy": ood_detection_accuracy,
            "mean_combined_fit_in_domain": avg(in_domain, "combined_fit"),
            "mean_combined_fit_ood": avg(ood, "combined_fit"),
            "note": "Fraction of ALL 52 queries where is_ood_predicted matched is_ood_ground_truth.",
        },

        "target_thresholds": {
            "mean_mrr": 0.80,
            "mean_recall_at_5": 0.90,
            "mean_precision_at_5": 0.20,
            "hit_rate_at_5": 0.95,
            "hit_rate_at_3": 0.85,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval eval set against the live pipeline.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N queries (use 3 for a quick sanity check before the full 52).")
    parser.add_argument("--debug", action="store_true",
                         help="Print raw chunk fields for the first result of each query, to verify category/title survive reranking.")
    args = parser.parse_args()

    print(f"Loading eval queries from: {EVAL_QUERIES_PATH}")
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    if args.limit:
        eval_queries = eval_queries[:args.limit]
        print(f"--limit {args.limit} applied -- running {len(eval_queries)} quer{'y' if len(eval_queries)==1 else 'ies'} only.")

    print(f"Loaded {len(eval_queries)} queries. Initializing retriever...")
    retriever = MultiQueryHybridRetriever()

    results = []
    for i, query_entry in enumerate(eval_queries, start=1):
        qid = query_entry["query_id"]
        print(f"[{i}/{len(eval_queries)}] Running {qid}: {query_entry['query'][:60]}...")

        start = time.perf_counter()
        result = run_single_query(retriever, query_entry, debug=args.debug)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        status = "ERROR" if result.get("error") else "ok"
        print(f"    -> mrr={result['mrr']} recall@5={result['recall_at_5']} "
              f"hit@5={result['hit_at_5']} action={result['action']} "
              f"combined_fit={result['combined_fit']} "
              f"({elapsed_ms}ms) [{status}]")

        results.append(result)

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = summarize(results)
    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVAL COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results -> {EVAL_RESULTS_PATH}")
    print(f"Summary          -> {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()