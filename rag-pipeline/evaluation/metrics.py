"""
Retrieval evaluation metrics.

Every function here takes:
  - retrieved_sources: list[str]  -> RANKED list, position 0 = top result
  - expected_sources:  list[str]  -> ground truth, unordered

And returns a float score. No side effects, no I/O, no dependency
on the retriever itself -- this is deliberate so it can be unit
tested with fake data before wiring into the real pipeline.
"""

from typing import List


def _normalize(source: str) -> str:
    """
    Strip file extensions and normalize slashes so 'agents/REACT.pdf'
    and 'agents/REACT' are treated as the same source.
    Adjust this if your actual chunk metadata format differs --
    this is the one place format mismatches get fixed centrally.
    """
    s = source.strip().replace("\\", "/")
    for ext in (".pdf", ".txt", ".md", ".docx"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s.lower()


def mean_reciprocal_rank(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """
    MRR (for a single query) = 1 / rank of the FIRST correct hit.
    Returns 0.0 if no expected source appears anywhere in retrieved_sources.
    """
    expected_set = {_normalize(s) for s in expected_sources}
    if not expected_set:
        # OOD queries have no expected sources -- MRR is not meaningful, return None-like 0.0
        return 0.0

    for rank, source in enumerate(retrieved_sources, start=1):
        if _normalize(source) in expected_set:
            return 1.0 / rank

    return 0.0


def recall_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int = 5) -> float:
    """
    Recall@k = (expected sources found in top-k) / (total expected sources).
    """
    expected_set = {_normalize(s) for s in expected_sources}
    if not expected_set:
        return 0.0

    top_k = {_normalize(s) for s in retrieved_sources[:k]}
    hits = expected_set & top_k  # set intersection
    return len(hits) / len(expected_set)


def precision_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int = 5) -> float:
    """
    Precision@k = (expected sources found in top-k) / k.
    Note: divides by k (slots used), not by len(retrieved) -- if the
    retriever returned fewer than k results, we still divide by k,
    penalizing under-filled results as intended.
    """
    expected_set = {_normalize(s) for s in expected_sources}
    if not expected_set or k == 0:
        return 0.0

    top_k = {_normalize(s) for s in retrieved_sources[:k]}
    hits = expected_set & top_k
    return len(hits) / k


def hit_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int = 5) -> bool:
    """
    Hit@k = was AT LEAST ONE expected source found anywhere in top-k?
    Binary, coarser signal than recall.
    """
    expected_set = {_normalize(s) for s in expected_sources}
    if not expected_set:
        return False

    top_k = {_normalize(s) for s in retrieved_sources[:k]}
    return len(expected_set & top_k) > 0


if __name__ == "__main__":
    # Quick self-test with fake data -- no retriever needed.
    # This is the "test each component in isolation" step.

    retrieved = [
        "rag/original_rag.pdf",
        "llm_engineering/flash_attention",
        "llm_engineering/attention_is_all_you_need",
        "deployment/vllm",
        "llm_engineering/lora",
    ]
    expected = ["llm_engineering/flash_attention"]

    print("=== Self-test: expected hit at rank 2 ===")
    print(f"MRR            : {mean_reciprocal_rank(retrieved, expected):.4f}  (expected 0.5)")
    print(f"Recall@5       : {recall_at_k(retrieved, expected, 5):.4f}  (expected 1.0)")
    print(f"Precision@5    : {precision_at_k(retrieved, expected, 5):.4f}  (expected 0.2)")
    print(f"Hit@3          : {hit_at_k(retrieved, expected, 3)}  (expected True)")
    print(f"Hit@1          : {hit_at_k(retrieved, expected, 1)}  (expected False)")

    print("\n=== Self-test: no hit at all ===")
    expected_miss = ["multi_agent/voyager"]
    print(f"MRR            : {mean_reciprocal_rank(retrieved, expected_miss):.4f}  (expected 0.0)")
    print(f"Recall@5       : {recall_at_k(retrieved, expected_miss, 5):.4f}  (expected 0.0)")

    print("\n=== Self-test: multiple expected sources (q017/q012 style) ===")
    expected_multi = ["rag/attention_is_all_you_need", "llm_engineering/attention_is_all_you_need"]
    print(f"MRR            : {mean_reciprocal_rank(retrieved, expected_multi):.4f}  (expected 0.333 - first hit at rank 3)")
    print(f"Recall@5       : {recall_at_k(retrieved, expected_multi, 5):.4f}  (expected 0.5 - only 1 of 2 found)")

    print("\n=== Self-test: OOD query (empty expected_sources) ===")
    print(f"MRR            : {mean_reciprocal_rank(retrieved, []):.4f}  (expected 0.0)")
    print(f"Hit@5          : {hit_at_k(retrieved, [], 5)}  (expected False)")