import numpy as np
import argparse
import sys
import re
from typing import Optional
from pathlib import Path

# Allow running this file directly from the retrieval folder.
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "rag-pipeline"

for path in (str(REPO_ROOT), str(PIPELINE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vectorstore.faiss_indexer import load_index
from ingestion.embedder import embed_chunks
from retrieval.sparse_retriever import SparseRetriever
from retrieval.query_rewriter import QueryRewriter

def _normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_weights(dense_weight, sparse_weight):
    total = max(dense_weight + sparse_weight, 1e-12)
    return dense_weight / total, sparse_weight / total


def _infer_weights(query, dense_weight, sparse_weight, adaptive_weights=False):
    if not adaptive_weights:
        return _normalize_weights(dense_weight, sparse_weight)

    q = query.lower()
    token_count = len(_tokenize(query))
    has_lexical_signals = (
        '"' in q
        or " and " in q
        or " or " in q
        or " not " in q
        or "exact" in q
        or "keyword" in q
    )

    if has_lexical_signals:
        return 0.4, 0.6
    if token_count >= 12:
        return 0.6, 0.4
    return 0.5, 0.5

class Retriever:
    def __init__(self, index_path=None):
        if index_path is None:
            index_path = str(PIPELINE_ROOT.parent / "data" / "processed")
        self.index, self.mapping = load_index(index_path)
        self.sparse = SparseRetriever(list(self.mapping.values()))


    def compare_query_rewrite(
            self,
            query,
            top_k=5,
            max_per_source: Optional[int] = 2,
            min_similarity=0.15,
            rewriter=None,
        ):
            if rewriter is None:
                rewriter = QueryRewriter()

            rewritten_query = rewriter.rewrite(query)

            dense_original = self.search(
                query,
                top_k=top_k,
                max_per_source=max_per_source,
                min_similarity=min_similarity,
            )

            dense_rewritten = self.search(
                rewritten_query,
                top_k=top_k,
                max_per_source=max_per_source,
                min_similarity=min_similarity,
            )

            sparse_original = self.sparse.search(query, top_k=top_k)
            sparse_rewritten = self.sparse.search(rewritten_query, top_k=top_k)

            return {
                "original_query": query,
                "rewritten_query": rewritten_query,
                "dense_original": dense_original,
                "dense_rewritten": dense_rewritten,
                "sparse_original": sparse_original,
                "sparse_rewritten": sparse_rewritten,
            }
    
    def compare_hybrid_rewrite(
        self,
        query,
        retrieve_k=80,
        final_k=30,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
        rewriter=None,
    ):
        if rewriter is None:
            rewriter = QueryRewriter()

        rewritten_query = rewriter.rewrite(query)

        original_out = self.hybrid_search_with_diagnostics(
            query,
            retrieve_k=retrieve_k,
            final_k=final_k,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=max_per_source,
        )

        rewritten_out = self.hybrid_search_with_diagnostics(
            rewritten_query,
            retrieve_k=retrieve_k,
            final_k=final_k,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=max_per_source,
        )

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "hybrid_original": original_out,
            "hybrid_rewritten": rewritten_out,
        }
    
    def _is_weak_candidate(self, chunk):
        text = chunk.get("text", "").strip().lower()

        if len(text) < 120:
            return True

        weak_patterns = [
            r"^references?$",
            r"^table of contents$",
            r"^page\s+\d+$",
            r"^figure\s+\d+.*$",
            r"^table\s+\d+.*$",
        ]

        for pattern in weak_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _filter_candidates(self, chunks, max_per_source: Optional[int] = 3):
        filtered = []
        per_source_counts = {}

        for chunk in chunks:
            if self._is_weak_candidate(chunk):
                continue

            source = chunk.get("source", "unknown")
            count = per_source_counts.get(source, 0)

            if max_per_source is not None and count >= max_per_source:
                continue

            per_source_counts[source] = count + 1
            filtered.append(chunk)

        return filtered
    
    #RETRIEVAL DIAGNOSTICS
    def _build_retrieval_diagnostics(
        self,
        query,
        expanded_queries,
        dense_results,
        sparse_results,
        final_results,
    ):
        diagnostics = {
            "query": query,
            "expanded_queries": expanded_queries,

            "dense_retrieval": {
                "count": len(dense_results),
                "sources": list(set(
                    chunk.get("source", "unknown")
                    for chunk in dense_results
                )),
                "top_scores": [
                    round(chunk.get("cosine_score", 0.0), 4)
                    for chunk in dense_results[:5]
                ],
            },

            "sparse_retrieval": {
                "count": len(sparse_results),
                "sources": list(set(
                    chunk.get("source", "unknown")
                    for chunk in sparse_results
                )),
                "top_scores": [
                    round(chunk.get("bm25_score", 0.0), 4)
                    for chunk in sparse_results[:5]
                ],
            },

            "final_results": {
                "count": len(final_results),
                "sources": list(set(
                    chunk.get("source", "unknown")
                    for chunk in final_results
                )),
                "top_rrf_scores": [
                    round(chunk.get("rrf_score", 0.0), 6)
                    for chunk in final_results[:5]
                ],
            },
        }

        return diagnostics


    def search(self, query, top_k=13, max_per_source: Optional[int] = 2, min_similarity=0.15):
        query_embedding = embed_chunks([{"text": query}])
        query_embedding = np.array(query_embedding).astype("float32")
        query_embedding = _normalize_embeddings(query_embedding)

        # Request more results from FAISS if max_per_source will filter some out
        # Heuristic: multiply by 3 to account for filtering overhead
        faiss_k = top_k if max_per_source is None else min(top_k * 3, self.index.ntotal)

        similarities, indices = self.index.search(query_embedding, faiss_k)
        results = []
        per_source_counts = {}

        for idx, similarity in zip(indices[0], similarities[0]):
            if len(results) >= top_k:
                break

            if similarity < min_similarity:
                continue

            if idx in self.mapping:
                # Copy chunk to avoid modifying original
                chunk = self.mapping[idx].copy()

                # Attach cosine similarity score (higher is better)
                chunk["cosine_score"] = float(similarity)

                if max_per_source is not None:
                    source = chunk.get("source", "unknown")
                    source_count = per_source_counts.get(source, 0)

                    if source_count >= max_per_source:
                        continue

                    per_source_counts[source] = source_count + 1

                results.append(chunk)

        results = self._filter_candidates(results, max_per_source=max_per_source)
        return results

    def hybrid_search(
        self,
        query,
        retrieve_k=80,
        final_k=30,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
    ):
        # Query Expansion: retrieve using both original and rewritten query
        queries = [query, QueryRewriter().rewrite(query)]
        dense_weight, sparse_weight = _infer_weights(
            query,
            dense_weight,
            sparse_weight,
            adaptive_weights=adaptive_weights,
        )

        all_dense = []
        all_sparse = []

        for q in queries:
            all_dense.extend(
                self.search(
                    q,
                    top_k=retrieve_k,
                    max_per_source=max_per_source,
                    min_similarity=min_dense_similarity,
                )
            )
            all_sparse.extend(self.sparse.search(q, top_k=retrieve_k, min_score=min_bm25_score))

        # --- NEW: dedupe before fusion/scoring ---
        def _dedupe_by_text(chunks):
            seen = set()
            unique = []
            for chunk in chunks:
                key = chunk["text"]
                if key not in seen:
                    unique.append(chunk)
                    seen.add(key)
            return unique

        all_dense = _dedupe_by_text(all_dense)
        all_sparse = _dedupe_by_text(all_sparse)

        all_dense = self._filter_candidates(all_dense, max_per_source=max_per_source)
        all_sparse = self._filter_candidates(all_sparse, max_per_source=max_per_source)
        # -----------------------------------------

        scores = {}

        # Dense ranking
        for rank, chunk in enumerate(all_dense):
            key = chunk["text"]
            scores[key] = scores.get(key, 0) + dense_weight * (1 / (k + rank + 1))

        # Sparse ranking
        for rank, chunk in enumerate(all_sparse):
            key = chunk["text"]
            scores[key] = scores.get(key, 0) + sparse_weight * (1 / (k + rank + 1))

        # Merge chunks
        combined = {}

        # Add dense first
        for chunk in all_dense:
            key = chunk["text"]
            combined[key] = chunk.copy()

        # Merge sparse WITHOUT overwriting
        for chunk in all_sparse:
            key = chunk["text"]
            if key not in combined:
                combined[key] = chunk.copy()

        # Attach score - RECIPROCAL RANKING FUSION (RRF) style (1 / (k + rank))
        for key in combined:
            combined[key]["rrf_score"] = scores.get(key, 0)

        # Sort
        final = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Light source balancing to reduce single-document dominance.
        source_limit = max_per_source if max_per_source is not None else final_k
        balanced = []
        per_source_counts = {}
        for chunk in final:
            source = chunk.get("source", "unknown")
            count = per_source_counts.get(source, 0)
            if count >= source_limit:
                continue
            per_source_counts[source] = count + 1
            balanced.append(chunk)
            if len(balanced) >= final_k:
                break

        return balanced

    def hybrid_search_with_diagnostics(
        self,
        query,
        retrieve_k=80,
        final_k=30,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
    ):
        queries = [query, QueryRewriter().rewrite(query)]
        dense_weight, sparse_weight = _infer_weights(
            query,
            dense_weight,
            sparse_weight,
            adaptive_weights=adaptive_weights,
        )

        all_dense = []
        all_sparse = []

        for q in queries:
            all_dense.extend(
                self.search(
                    q,
                    top_k=retrieve_k,
                    max_per_source=max_per_source,
                    min_similarity=min_dense_similarity,
                )
            )
            all_sparse.extend(
                self.sparse.search(q, top_k=retrieve_k, min_score=min_bm25_score)
            )

        def _dedupe_by_text(chunks):
            seen = set()
            unique = []
            for chunk in chunks:
                key = chunk["text"]
                if key not in seen:
                    unique.append(chunk)
                    seen.add(key)
            return unique

        all_dense = _dedupe_by_text(all_dense)
        all_sparse = _dedupe_by_text(all_sparse)

        all_dense = self._filter_candidates(all_dense, max_per_source=max_per_source)
        all_sparse = self._filter_candidates(all_sparse, max_per_source=max_per_source)

        scores = {}

        for rank, chunk in enumerate(all_dense):
            key = chunk["text"]
            scores[key] = scores.get(key, 0) + dense_weight * (1 / (k + rank + 1))

        for rank, chunk in enumerate(all_sparse):
            key = chunk["text"]
            scores[key] = scores.get(key, 0) + sparse_weight * (1 / (k + rank + 1))

        combined = {}

        for chunk in all_dense:
            key = chunk["text"]
            combined[key] = chunk.copy()

        for chunk in all_sparse:
            key = chunk["text"]
            if key not in combined:
                combined[key] = chunk.copy()

        for key in combined:
            combined[key]["rrf_score"] = scores.get(key, 0)

        final = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

        source_limit = max_per_source if max_per_source is not None else final_k
        balanced = []
        per_source_counts = {}
        for chunk in final:
            source = chunk.get("source", "unknown")
            count = per_source_counts.get(source, 0)
            if count >= source_limit:
                continue
            per_source_counts[source] = count + 1
            balanced.append(chunk)
            if len(balanced) >= final_k:
                break

        diagnostics = self._build_retrieval_diagnostics(
            query=query,
            expanded_queries=queries,
            dense_results=all_dense,
            sparse_results=all_sparse,
            final_results=balanced,
        )

        return {
            "results": balanced,
            "diagnostics": diagnostics,
        }


def print_diagnostics(label, diag):
    print(f"\n================ DIAGNOSTICS ({label}) ================\n")
    print(f"Query: {diag['query']}")

    print("\nExpanded queries:")
    for i, q in enumerate(diag["expanded_queries"], start=1):
        print(f"  [{i}] {q}")

    print("\nDense retrieval:")
    print(f"  Count: {diag['dense_retrieval']['count']}")
    print(f"  Sources: {', '.join(diag['dense_retrieval']['sources'])}")
    print(f"  Top scores: {diag['dense_retrieval']['top_scores']}")

    print("\nSparse retrieval:")
    print(f"  Count: {diag['sparse_retrieval']['count']}")
    print(f"  Sources: {', '.join(diag['sparse_retrieval']['sources'])}")
    print(f"  Top scores: {diag['sparse_retrieval']['top_scores']}")

    print("\nFinal results:")
    print(f"  Count: {diag['final_results']['count']}")
    print(f"  Sources: {', '.join(diag['final_results']['sources'])}")
    print(f"  Top RRF scores: {diag['final_results']['top_rrf_scores']}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test dense/hybrid retriever with a query.")
    # parser.add_argument("query", nargs="?", default="What is dense retrieval?", help="Query text")
    # parser.add_argument(
    #     "--mode",
    #     choices=["dense", "hybrid"],
    #     default="hybrid",
    #     help="Retrieval mode: dense only or hybrid (dense + BM25)",
    # )
    # parser.add_argument("--retrieve-k", "--retrieve_k", type=int, default=50, help="Number of chunks to retrieve first")
    # parser.add_argument("--final-k", "--final_k", type=int, default=20, help="Number of final results to return")
    # parser.add_argument("--dense-weight", type=float, default=0.5, help="Dense fusion weight for hybrid mode")
    # parser.add_argument("--sparse-weight", type=float, default=0.5, help="Sparse fusion weight for hybrid mode")
    # parser.add_argument(
    #     "--adaptive-weights",
    #     action="store_true",
    #     help="Use simple query-aware adaptive weighting instead of fixed dense/sparse weights",
    # )
    # parser.add_argument(
    #     "--min-dense-similarity",
    #     type=float,
    #     default=0.15,
    #     help="Minimum dense cosine similarity to keep a chunk",
    # )
    # parser.add_argument(
    #     "--min-bm25-score",
    #     type=float,
    #     default=0.0,
    #     help="Minimum BM25 score to keep a sparse chunk in hybrid mode",
    # )
    # parser.add_argument(
    #     "--preview-chars",
    #     type=int,
    #     default=220,
    #     help="Number of characters to print per retrieved chunk preview",
    # )
    # parser.add_argument(
    #     "--full-text",
    #     action="store_true",
    #     help="Print full chunk text instead of truncating",
    # )
    # parser.add_argument(
    #     "--max-per-source",
    #     type=int,
    #     default=3,
    #     help="Cap on chunks returned per source document (set 0 or negative to disable)",
    # )
    # parser.add_argument(
    #     "--index-path",
    #     default=str(PROJECT_ROOT / "data" / "processed"),
    #     help="Path containing faiss_index.bin and id_to_chunk.pkl",
    # )
    # args = parser.parse_args()

    # retriever = Retriever(index_path=args.index_path)
    # max_per_source = args.max_per_source if args.max_per_source and args.max_per_source > 0 else None
    # if args.mode == "hybrid":
    #     results = retriever.hybrid_search(
    #         args.query,
    #         retrieve_k=args.retrieve_k,
    #         final_k=args.final_k,
    #         dense_weight=args.dense_weight,
    #         sparse_weight=args.sparse_weight,
    #         adaptive_weights=args.adaptive_weights,
    #         min_dense_similarity=args.min_dense_similarity,
    #         min_bm25_score=args.min_bm25_score,
    #         max_per_source=max_per_source,
    #     )
    # else:
    #     results = retriever.search(
    #         args.query,
    #         top_k=args.retrieve_k,
    #         max_per_source=max_per_source,
    #         min_similarity=args.min_dense_similarity,
    #     )

    # print(f"Query: {args.query}")
    # print(f"Mode: {args.mode}")
    # if args.mode == "hybrid":
    #     print(f"Retrieve-k: {args.retrieve_k}")
    #     print(f"Final-k: {args.final_k}")
    # else:
    #     print(f"Top-k: {args.retrieve_k}")
    # print(f"Results: {len(results)}")

    # if results:
    #     highest_cosine = max(item.get("cosine_score", float("-inf")) for item in results)
    #     print(f"Highest cosine similarity in retrieved set: {highest_cosine:.4f} (higher is better)")

    # def format_chunk_text(text):
    #     clean_text = text.replace("\n", " ")
    #     if args.full_text:
    #         return clean_text
    #     return clean_text[:args.preview_chars]

    # if not results:
    #     print("No results found.")
    # else:
    #     for i, item in enumerate(results, start=1):
    #         source = item.get("source", "unknown")
    #         cosine_score = item.get("cosine_score", float("nan"))
    #         text = item.get("text", "")
    #         print(f"\n[{i}] cosine_score={cosine_score:.4f} source={source}")
    #         print(format_chunk_text(text))

    retriever = Retriever()
    rewriter = QueryRewriter()

    query = "What is dense and sparse retrieval?"

    output = retriever.compare_hybrid_rewrite(query, final_k=10, rewriter=rewriter)

    print("\n================ ORIGINAL QUERY ================\n")
    print(output["original_query"])

    print("\n================ REWRITTEN QUERY ================\n")
    print(output["rewritten_query"])

    print_diagnostics("ORIGINAL", output["hybrid_original"]["diagnostics"])
    print_diagnostics("REWRITTEN", output["hybrid_rewritten"]["diagnostics"])

    print("\n================ HYBRID RESULTS (ORIGINAL) ================\n")
    original_results = output["hybrid_original"]["results"]
    for i, c in enumerate(original_results, start=1):
        print(f"[{i}] source={c.get('source', 'unknown')}")
        print(c.get("text", "")[:300])
        print("-" * 80)

    print("\n================ HYBRID RESULTS (REWRITTEN) ================\n")
    rewritten_results = output["hybrid_rewritten"]["results"]
    for i, c in enumerate(rewritten_results, start=1):
        print(f"[{i}] source={c.get('source', 'unknown')}")
        print(c.get("text", "")[:300])
        print("-" * 80)