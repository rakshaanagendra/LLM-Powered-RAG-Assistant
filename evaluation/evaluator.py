import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval.retriever import Retriever
from rag.retrieval.reranker import Reranker
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_queries():
    path = PROJECT_ROOT / "data" / "queries" / "eval_queries.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate():
    retriever = Retriever()
    reranker = Reranker()

    queries = load_queries()

    top1 = 0
    top3 = 0
    total = len(queries)

    for item in queries:
        query = item["question"]
        expected = item["expected_doc"]

        retrieved = retriever.hybrid_search(query, retrieve_k=80, final_k=30)
        reranked = reranker.rerank(query, retrieved, top_k=3)

        # Extract sources
        results = [r.get("source", "").lower() for r in reranked]

        # Normalize expected_doc name
        expected = expected.lower()

        # Top-1
        if results and expected in results[0]:
            top1 += 1

        # Top-3
        if any(expected in r for r in results):
            top3 += 1

        print("\n----------------------------")
        print("Query:", query)
        print("Expected:", expected)
        print("Got:", results)

    print("\n========== RESULTS ==========")
    print(f"Top-1 Accuracy: {top1}/{total} = {top1/total:.2f}")
    print(f"Top-3 Accuracy: {top3}/{total} = {top3/total:.2f}")

if __name__ == "__main__":
    evaluate()