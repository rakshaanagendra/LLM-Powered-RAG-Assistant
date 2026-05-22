from typing import Optional, List, Dict, Any
import io
import sys
from pathlib import Path
from collections import Counter
import re


if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT)) 

from retrieval.retriever import Retriever
from retrieval.multi_query_retriever import MultiQueryRetriever
from retrieval.reranker import Reranker


class MultiQueryHybridRetriever:
    def __init__(self, index_path=None, model_name="qwen2.5:3b", client=None):
        self.retriever = Retriever(index_path=index_path)
        self.query_generator = MultiQueryRetriever(
            model_name=model_name,
            client=client,
        )
        self.reranker = Reranker()

    def _deduplicate_by_text(self, chunks):
        seen = set()
        unique = []

        for chunk in chunks:
            text = chunk.get("text", "")
            key = text.strip()

            if key and key not in seen:
                unique.append(chunk)
                seen.add(key)

        return unique

    def _aggregate_results(self, query_results, k=60, max_per_source: Optional[int] = 3):
        scores = {}
        combined = {}

        for query_rank, chunks in enumerate(query_results):
            for rank, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                key = text.strip()

                if not key:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if key not in combined:
                    combined[key] = chunk.copy()

                combined[key]["rrf_score"] = combined[key].get("rrf_score", 0.0) + rrf_score
                scores[key] = scores.get(key, 0.0) + rrf_score

        for key in combined:
            combined[key]["multi_query_rrf_score"] = scores.get(key, 0.0)

        ranked = sorted(
            combined.values(),
            key=lambda x: x["multi_query_rrf_score"],
            reverse=True,
        )

        if max_per_source is None:
            return ranked

        balanced = []
        per_source_counts = {}

        for chunk in ranked:
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            count = per_source_counts.get(source, 0)

            if count >= max_per_source:
                continue

            per_source_counts[source] = count + 1
            balanced.append(chunk)

        return balanced

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        include_scores: bool = True,
    ) -> str:
        """
        Convert the top reranked chunks into a clean prompt-ready context string.
        """
        if not chunks:
            return ""

        selected_chunks = chunks[:top_k]
        context_parts = []

        for i, chunk in enumerate(selected_chunks, start=1):
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            page = chunk.get("page") or chunk.get("metadata", {}).get("page") or "unknown"
            text = chunk.get("text", "").strip()

            if not text:
                continue

            header_lines = [f"[Chunk {i}]"]
            header_lines.append(f"Source: {source}")
            header_lines.append(f"Page: {page}")

            if include_scores:
                rerank_score = chunk.get("rerank_score", None)
                rrf_score = chunk.get("multi_query_rrf_score", None)

                if rerank_score is not None:
                    header_lines.append(f"Rerank Score: {rerank_score:.4f}")
                if rrf_score is not None:
                    header_lines.append(f"RRF Score: {rrf_score:.4f}")

            header = "\n".join(header_lines)
            context_parts.append(f"{header}\nText:\n{text}")

        return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(context_parts) if context_parts else ""


    def search(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
    ):
        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
        )

        query_results = []

        for q in expanded_queries:
            results = self.retriever.hybrid_search(
                q,
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
            query_results.append(results)

        merged = self._aggregate_results(
            query_results,
            k=k,
            max_per_source=max_per_source,
        )

        # Rerank merged results with Reranker
        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=merged,
            top_k=final_k,
        )

        return reranked

    def search_with_context(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
        context_top_k: int = 5,
        include_scores: bool = True,
    ):
        """
        Returns both the reranked chunks and a prompt-ready context string.
        """
        # Generate expanded queries (useful to return to callers)
        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
        )

        # Perform the multi-query search and rerank
        reranked_chunks = self.search(
            query=query,
            num_queries=num_queries,
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

        # Build the prompt-ready context string and also expose the chunks used
        context = self.build_context(
            reranked_chunks,
            top_k=context_top_k,
            include_scores=include_scores,
        )

        context_chunks = reranked_chunks[:context_top_k]

        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "reranked_chunks": reranked_chunks,
            "context_chunks": context_chunks,
            "context": context,
        }

    def search_with_diagnostics(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
        context_top_k: int = 5,
        include_scores: bool = True,
    ):
        """
        Multi-query hybrid retrieval with structured diagnostics.

        Returns:
            {
                "query": ...,
                "expanded_queries": ...,
                "per_query_results": ...,
                "merged_chunks": ...,
                "reranked_chunks": ...,
                "context_chunks": ...,
                "context": ...,
                "diagnostics": ...
            }
        """

        def _chunk_source(chunk):
            return chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"

        def _summarize_chunks(chunks, score_key=None, top_n=5):
            return {
                "count": len(chunks),
                "sources": sorted({_chunk_source(c) for c in chunks}),
                "source_distribution": dict(Counter(_chunk_source(c) for c in chunks)),
                "top_scores": [
                    round(c.get(score_key, 0.0), 6) if score_key else None
                    for c in chunks[:top_n]
                ] if chunks else [],
            }

        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
        )

        per_query_results = []

        for expanded_query in expanded_queries:
            results = self.retriever.hybrid_search(
                expanded_query,
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

            per_query_results.append(
                {
                    "query": expanded_query,
                    "results": results,
                    "summary": {
                        "count": len(results),
                        "sources": sorted({_chunk_source(c) for c in results}),
                        "source_distribution": dict(Counter(_chunk_source(c) for c in results)),
                        "top_cosine_scores": [
                            round(c.get("cosine_score", 0.0), 4)
                            for c in results[:5]
                        ],
                        "top_rrf_scores": [
                            round(c.get("rrf_score", 0.0), 6)
                            for c in results[:5]
                        ],
                    },
                }
            )

        query_result_lists = [entry["results"] for entry in per_query_results]

        def _normalize_text(text: str) -> str:
            return re.sub(r"\s+", " ", text.strip().lower())

        def _compute_duplicate_stats(query_results):
            all_chunks = [
                chunk
                for results in query_results
                for chunk in results
                if chunk.get("text", "").strip()
            ]

            total_candidates = len(all_chunks)

            if total_candidates == 0:
                return {
                    "total_candidates": 0,
                    "unique_candidates": 0,
                    "duplicate_candidates": 0,
                    "duplicate_rate": 0.0,
                }

            seen = set()
            duplicate_candidates = 0

            for chunk in all_chunks:
                key = _normalize_text(chunk["text"])
                if key in seen:
                    duplicate_candidates += 1
                else:
                    seen.add(key)

            unique_candidates = total_candidates - duplicate_candidates
            duplicate_rate = duplicate_candidates / total_candidates

            return {
                "total_candidates": total_candidates,
                "unique_candidates": unique_candidates,
                "duplicate_candidates": duplicate_candidates,
                "duplicate_rate": round(duplicate_rate, 4),
            }

        duplicate_stats = _compute_duplicate_stats(query_result_lists)

        def _compute_metadata_coverage(chunks, keys=None):
            if keys is None:
                keys = ["source", "source_path", "doc_type", "doc_id", "chunk_id", "page", "title"]

            total = len(chunks)
            coverage = {}

            for key in keys:
                present = 0
                for chunk in chunks:
                    value = chunk.get(key, None)
                    if value not in (None, "", "unknown"):
                        present += 1

                coverage[key] = {
                    "present": present,
                    "total": total,
                    "coverage_rate": round((present / total), 4) if total else 0.0,
                }

            return coverage

        def _compute_source_diversity(chunks, top_k=3):
            total = len(chunks)

            if total == 0:
                return {
                    "total_chunks": 0,
                    "unique_sources": 0,
                    "dominant_source": None,
                    "dominant_source_count": 0,
                    "dominant_ratio": 0.0,
                    "top_k_sources": [],
                    "top_k_count": 0,
                    "top_k_ratio": 0.0,
                    "source_distribution": {},
                    "balance": "empty",
                }

            source_counts = Counter(_chunk_source(c) for c in chunks)
            dominant_source, dominant_count = source_counts.most_common(1)[0]

            top_k_items = source_counts.most_common(top_k)
            top_k_count = sum(count for _, count in top_k_items)
            top_k_ratio = top_k_count / total

            dominant_ratio = dominant_count / total

            if dominant_ratio <= 0.34:
                balance = "good"
            elif dominant_ratio <= 0.60:
                balance = "moderate"
            else:
                balance = "poor"

            return {
                "total_chunks": total,
                "unique_sources": len(source_counts),
                "dominant_source": dominant_source,
                "dominant_source_count": dominant_count,
                "dominant_ratio": round(dominant_ratio, 4),
                "top_k_sources": top_k_items,
                "top_k_count": top_k_count,
                "top_k_ratio": round(top_k_ratio, 4),
                "source_distribution": dict(source_counts),
                "balance": balance,
            }

        def _compute_rerank_lift(before_chunks, after_chunks):
            def _normalize_text(text: str) -> str:
                return re.sub(r"\s+", " ", text.strip().lower())

            # Rank in the merged list before reranking
            before_rank = {}
            for idx, chunk in enumerate(before_chunks, start=1):
                key = _normalize_text(chunk.get("text", ""))
                if key and key not in before_rank:
                    before_rank[key] = idx

            # Rank in the reranked list after reranking
            after_rank = {}
            after_chunk_map = {}
            for idx, chunk in enumerate(after_chunks, start=1):
                key = _normalize_text(chunk.get("text", ""))
                if key and key not in after_rank:
                    after_rank[key] = idx
                    after_chunk_map[key] = chunk

            rows = []
            for key, b_rank in before_rank.items():
                a_rank = after_rank.get(key)
                if a_rank is None:
                    continue

                chunk = after_chunk_map[key]
                rows.append({
                    "source": chunk.get("source", "unknown"),
                    "before_rank": b_rank,
                    "after_rank": a_rank,
                    "lift": b_rank - a_rank,
                    "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
                    "rrf_score": round(chunk.get("multi_query_rrf_score", 0.0), 6),
                    "text_preview": chunk.get("text", "")[:120].replace("\n", " "),
                })

            if not rows:
                return {
                    "compared_candidates": 0,
                    "average_lift": 0.0,
                    "average_abs_lift": 0.0,
                    "promoted": 0,
                    "demoted": 0,
                    "unchanged": 0,
                    "top_gains": [],
                    "top_losses": [],
                }

            promoted = [r for r in rows if r["lift"] > 0]
            demoted = [r for r in rows if r["lift"] < 0]
            unchanged = [r for r in rows if r["lift"] == 0]

            top_gains = sorted(rows, key=lambda r: r["lift"], reverse=True)[:5]
            top_losses = sorted(rows, key=lambda r: r["lift"])[:5]

            return {
                "compared_candidates": len(rows),
                "average_lift": round(sum(r["lift"] for r in rows) / len(rows), 2),
                "average_abs_lift": round(sum(abs(r["lift"]) for r in rows) / len(rows), 2),
                "promoted": len(promoted),
                "demoted": len(demoted),
                "unchanged": len(unchanged),
                "top_gains": top_gains,
                "top_losses": top_losses,
            }

        merged = self._aggregate_results(
            query_result_lists,
            k=k,
            max_per_source=max_per_source,
)

        # Keep a copy before reranking because reranker mutates chunks in place.
        merged_before_rerank = [chunk.copy() for chunk in merged]

        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=merged,
            top_k=final_k,
        )

        rerank_lift = _compute_rerank_lift(merged_before_rerank, reranked)

        context = self.build_context(
            reranked,
            top_k=context_top_k,
            include_scores=include_scores,
        )

        context_chunks = reranked[:context_top_k]

        metadata_coverage = {
        "merged": _compute_metadata_coverage(merged_before_rerank),
        "reranked": _compute_metadata_coverage(reranked),
        "context": _compute_metadata_coverage(context_chunks),
    }
        
        source_diversity = {
        "merged": _compute_source_diversity(merged_before_rerank),
        "reranked": _compute_source_diversity(reranked),
        "context": _compute_source_diversity(context_chunks),
    }

        diagnostics = {
            "query": query,
            "expanded_queries": expanded_queries,
            "num_expanded_queries": len(expanded_queries),
            "duplicate_stats": duplicate_stats,
            "rerank_lift": rerank_lift,
            "metadata_coverage": metadata_coverage,
            "source_diversity": source_diversity,
            "per_query_results": [
                {
                    "query": item["query"],
                    "count": item["summary"]["count"],
                    "sources": item["summary"]["sources"],
                    "source_distribution": item["summary"]["source_distribution"],
                    "top_cosine_scores": item["summary"]["top_cosine_scores"],
                    "top_rrf_scores": item["summary"]["top_rrf_scores"],
                }
                for item in per_query_results
            ],
            "merged_results": {
                **_summarize_chunks(merged_before_rerank),
                "top_multi_query_rrf_scores": [
                    round(c.get("multi_query_rrf_score", 0.0), 6)
                    for c in merged_before_rerank[:5]
                ],
            },
            "reranked_results": {
                **_summarize_chunks(reranked, score_key="rerank_score"),
                "top_rerank_scores": [
                    round(c.get("rerank_score", 0.0), 4)
                    for c in reranked[:5]
                ],
            },
            "context_summary": {
                "count": len(context_chunks),
                "sources": sorted({_chunk_source(c) for c in context_chunks}),
                "source_distribution": dict(Counter(_chunk_source(c) for c in context_chunks)),
            },
        }

        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "per_query_results": per_query_results,
            "merged_chunks": merged_before_rerank,
            "reranked_chunks": reranked,
            "context_chunks": context_chunks,
            "context": context,
            "diagnostics": diagnostics,
        }

        

    def print_diagnostics(self, diag):
        print("\n==================================================")
        print("RETRIEVAL DIAGNOSTICS")
        print("==================================================")

        print("\nORIGINAL QUERY:")
        print(diag["query"])

        print("\nEXPANDED QUERIES:")
        for i, q in enumerate(diag["expanded_queries"], start=1):
            print(f"  [{i}] {q}")

        print("\nDUPLICATE STATS:")
        dup = diag["duplicate_stats"]
        print(f"  Total candidates: {dup['total_candidates']}")
        print(f"  Unique candidates: {dup['unique_candidates']}")
        print(f"  Duplicate candidates: {dup['duplicate_candidates']}")
        print(f"  Duplicate rate: {dup['duplicate_rate']:.4f}")

        print("\nRERANK LIFT:")
        lift = diag["rerank_lift"]
        print(f"  Compared candidates: {lift['compared_candidates']}")
        print(f"  Average lift: {lift['average_lift']:.2f}")
        print(f"  Average absolute lift: {lift['average_abs_lift']:.2f}")
        print(f"  Promoted: {lift['promoted']}")
        print(f"  Demoted: {lift['demoted']}")
        print(f"  Unchanged: {lift['unchanged']}")

        print("\nTOP GAINS:")
        if lift["top_gains"]:
            for idx, item in enumerate(lift["top_gains"], start=1):
                print(f"\n[GAIN {idx}]")
                print(f"Source: {item['source']}")
                print(f"Before Rank: {item['before_rank']}")
                print(f"After Rank: {item['after_rank']}")
                print(f"Lift: +{item['lift']}")
                print(f"Rerank Score: {item['rerank_score']:.4f}")
                print(f"RRF Score: {item['rrf_score']:.6f}")
                print("Preview:")
                print(item["text_preview"])
                print("-" * 60)
        else:
            print("None")

        print("\nTOP LOSSES:")
        if lift["top_losses"]:
            for idx, item in enumerate(lift["top_losses"], start=1):
                print(f"\n[LOSS {idx}]")
                print(f"Source: {item['source']}")
                print(f"Before Rank: {item['before_rank']}")
                print(f"After Rank: {item['after_rank']}")
                print(f"Lift: {item['lift']}")
                print(f"Rerank Score: {item['rerank_score']:.4f}")
                print(f"RRF Score: {item['rrf_score']:.6f}")
                print("Preview:")
                print(item["text_preview"])
                print("-" * 60)
        else:
            print("None")

        print("\nMETADATA COVERAGE:")

        for stage_name, stage_cov in diag["metadata_coverage"].items():
            print(f"\n{stage_name.upper()} CHUNKS:")
            for field, stats in stage_cov.items():
                print(
                    f"  {field}: {stats['present']}/{stats['total']} "
                    f"({stats['coverage_rate']:.4f})"
                )

        print("\nSOURCE DIVERSITY:")

        for stage_name, stage in diag["source_diversity"].items():
            print(f"\n{stage_name.upper()} CHUNKS:")
            print(f"  Total chunks: {stage['total_chunks']}")
            print(f"  Unique sources: {stage['unique_sources']}")
            print(f"  Dominant source: {stage['dominant_source']}")
            print(f"  Dominant source count: {stage['dominant_source_count']}")
            print(f"  Dominant ratio: {stage['dominant_ratio']:.4f}")
            print(f"  Top-3 source count: {stage['top_k_count']}")
            print(f"  Top-3 source ratio: {stage['top_k_ratio']:.4f}")
            print(f"  Balance: {stage['balance']}")

            print("  Source distribution:")
            for source, count in stage["source_distribution"].items():
                print(f"    {source}: {count}")

        print("\n==================================================")
        print("PER-QUERY RETRIEVAL")
        print("==================================================")

        for i, item in enumerate(diag["per_query_results"], start=1):
            print(f"\nQUERY VARIANT [{i}]")
            print(f"Query: {item['query']}")
            print(f"Retrieved Count: {item['count']}")

            print("\nSources:")
            for s in item["sources"]:
                print(f"  - {s}")

            print("\nSource Distribution:")
            for source, count in item["source_distribution"].items():
                print(f"  {source}: {count}")

            print(f"\nTop Cosine Scores: {item['top_cosine_scores']}")
            print(f"Top RRF Scores: {item['top_rrf_scores']}")

        print("\n==================================================")
        print("MERGED RESULTS")
        print("==================================================")

        merged = diag["merged_results"]

        print(f"Merged Candidate Count: {merged['count']}")

        print("\nMerged Source Distribution:")
        for source, count in merged["source_distribution"].items():
            print(f"  {source}: {count}")

        print(f"\nTop Multi-Query RRF Scores:")
        print(merged["top_multi_query_rrf_scores"])

        print("\n==================================================")
        print("RERANKED RESULTS")
        print("==================================================")

        reranked = diag["reranked_results"]

        print(f"Reranked Count: {reranked['count']}")

        print("\nReranked Source Distribution:")
        for source, count in reranked["source_distribution"].items():
            print(f"  {source}: {count}")

        print(f"\nTop Rerank Scores:")
        print(reranked["top_rerank_scores"])

        print("\n==================================================")
        print("FINAL CONTEXT SUMMARY")
        print("==================================================")

        context = diag["context_summary"]

        print(f"Context Chunk Count: {context['count']}")

        print("\nContext Sources:")
        for source, count in context["source_distribution"].items():
            print(f"  {source}: {count}")

## Basic test cases for MultiQueryHybridRetriever
# if __name__ == "__main__":
#     multi_query_retriever = MultiQueryHybridRetriever()

#     test_queries = [
#         "What is dense and sparse retrieval?",
#         "that thing where llms forget the middle part",
#         "how do vector databases work",
#         "bm25 vs embeddings",
#     ]

#     for query in test_queries:
#         print("\n==================================================")
#         print("ORIGINAL QUERY:")
#         print(query)

#         expanded_queries = multi_query_retriever.query_generator.generate_queries(
#             query,
#             num_queries=4,
#         )

#         print("\nEXPANDED QUERIES:\n")

#         for i, q in enumerate(expanded_queries, start=1):
#             print(f"[{i}] {q}")

#         results = multi_query_retriever.search(
#             query,
#             num_queries=4,
#             retrieve_k=40,
#             final_k=10,
#         )

#         print("\nFINAL RERANKED RESULTS:\n")

#         for i, c in enumerate(results, start=1):
#             print(f"[{i}] source={c.get('source', 'unknown')}")

#             if "rerank_score" in c:
#                 print(f"rerank_score={c['rerank_score']:.4f}")

#             if "multi_query_rrf_score" in c:
#                 print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")

#             print(c.get("text", "")[:300])

#             print("-" * 80)

## Additional test case for search_with_context i.e. retrieving results along with a prompt-ready context string
if __name__ == "__main__":
    multi_query_retriever = MultiQueryHybridRetriever()

    test_queries = [
        "What is dense and sparse retrieval?",
        "that thing where llms forget the middle part",
        "how do vector databases work",
        "bm25 vs embeddings",
    ]

    for query in test_queries:
        print("\n==================================================")
        print("ORIGINAL QUERY:")
        print(query)

        expanded_queries = multi_query_retriever.query_generator.generate_queries(
            query,
            num_queries=4,
        )

        print("\nEXPANDED QUERIES:\n")

        for i, q in enumerate(expanded_queries, start=1):
            print(f"[{i}] {q}")

        result = multi_query_retriever.search_with_diagnostics(
            query,
            num_queries=4,
            retrieve_k=40,
            final_k=10,
            context_top_k=5,
        )

        multi_query_retriever.print_diagnostics(result["diagnostics"])

        print("\n==================================================")
        print("FINAL RERANKED RESULTS")
        print("==================================================")

        for i, c in enumerate(result["reranked_chunks"], start=1):
            print(f"\n[{i}] source={c.get('source', 'unknown')}")

            if "rerank_score" in c:
                print(f"rerank_score={c['rerank_score']:.4f}")

            if "multi_query_rrf_score" in c:
                print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")

            print(c.get("text", "")[:300])

            print("-" * 80)

        print("\n==================================================")
        print("PROMPT-READY CONTEXT")
        print("==================================================")

        print(result["context"])