from typing import Dict, Any, List, Optional
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
from generation.generator import Generator


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

    # After reranking, we may want to enforce a hard cap on how many chunks from the same source can appear in the final context.
    def _apply_source_cap(self, chunks, max_per_source: Optional[int]):
        if max_per_source is None:
            return chunks

        selected = []
        counts = {}

        for chunk in chunks:
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            count = counts.get(source, 0)

            if count >= max_per_source:
                continue

            counts[source] = count + 1
            selected.append(chunk)

        return selected

    def _compute_retrieval_health(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine retrieval diagnostics into one health decision.
        Returns:
            {
                "is_weak": bool,
                "severity": "strong" | "medium" | "weak",
                "risk_score": float,
                "reasons": list[str],
                "summary": dict
            }
        """
        rerank_scores = diagnostics.get("rerank_scores", [])
        duplicate_rate = diagnostics.get("duplicate_rate", 0.0)
        source_diversity = diagnostics.get("source_diversity", 0.0)
        dominant_source_ratio = diagnostics.get("dominant_source_ratio", 0.0)
        top3_source_concentration = diagnostics.get("top3_source_concentration", 0.0)
        lexical_overlap = diagnostics.get("avg_lexical_overlap", 1.0)

        if not rerank_scores:
            return {
                "is_weak": True,
                "severity": "weak",
                "risk_score": 1.0,
                "reasons": ["No rerank scores were produced."],
                "summary": {},
            }

        avg_score = sum(rerank_scores) / len(rerank_scores)
        top_score = max(rerank_scores)

        if len(rerank_scores) > 1:
            sorted_scores = sorted(rerank_scores, reverse=True)
            score_gap = sorted_scores[0] - sorted_scores[1]
            mean_score = avg_score
            variance = sum((s - mean_score) ** 2 for s in rerank_scores) / len(rerank_scores)
            score_std = variance ** 0.5
        else:
            score_gap = top_score
            score_std = 0.0

        reasons = []
        risk_score = 0.0

        if avg_score < 0.35:
            risk_score += 0.30
            reasons.append(f"Average rerank score is low ({avg_score:.3f}).")

        if top_score < 0.50:
            risk_score += 0.25
            reasons.append(f"Top rerank score is weak ({top_score:.3f}).")

        if score_gap < 0.05:
            risk_score += 0.15
            reasons.append(f"Rerank scores are flat (top gap {score_gap:.3f}).")

        if score_std < 0.03:
            risk_score += 0.10
            reasons.append(f"Rerank scores have very low spread (std {score_std:.3f}).")

        if duplicate_rate > 0.40:
            risk_score += 0.15
            reasons.append(f"Duplicate rate is high ({duplicate_rate:.2%}).")

        if source_diversity < 0.40:
            risk_score += 0.10
            reasons.append(f"Source diversity is low ({source_diversity:.3f}).")

        if dominant_source_ratio > 0.70:
            risk_score += 0.10
            reasons.append(f"One source dominates too much ({dominant_source_ratio:.3f}).")

        if top3_source_concentration > 0.85:
            risk_score += 0.10
            reasons.append(f"Top-3 sources dominate the context ({top3_source_concentration:.3f}).")

        if lexical_overlap < 0.20:
            risk_score += 0.15
            reasons.append(f"Lexical overlap is weak ({lexical_overlap:.3f}).")

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.60:
            severity = "weak"
            is_weak = True
        elif risk_score >= 0.30:
            severity = "medium"
            is_weak = False
        else:
            severity = "strong"
            is_weak = False

        return {
            "is_weak": is_weak,
            "severity": severity,
            "risk_score": round(risk_score, 3),
            "reasons": reasons,
            "summary": {
                "avg_score": round(avg_score, 3),
                "top_score": round(top_score, 3),
                "score_gap": round(score_gap, 3),
                "score_std": round(score_std, 3),
            },
        }


    # For specific technical queries, we may want a more targeted confidence measure that focuses on relevance signals rather than diversity or duplication, since for some questions it's more important to find the right piece of evidence even if it's from the same source.
    # So, a practical rule would be to separate relevance score and diversity score so that a low diversity score does not overpower a strong match
    # For relevance - top rerank score, average rerank score, score gap and lexical overlap
    # For diversity - duplicate rate, source diversity, dominant source ratio, top3 source concentration
    def _compute_relevance_confidence(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measures how relevant the retrieved chunks look.
        This should answer: 'Did I find the right stuff?'
        """
        rerank_scores = diagnostics.get("rerank_scores", [])
        lexical_overlap = diagnostics.get("avg_lexical_overlap", 0.0)

        if not rerank_scores:
            return {
                "score": 0.0,
                "label": "weak",
                "reason": "No rerank scores available.",
            }

        avg_score = sum(rerank_scores) / len(rerank_scores)
        top_score = max(rerank_scores)

        # Simple relevance score: emphasize top rerank score + average score + lexical overlap
        score = (
            0.45 * top_score +
            0.35 * avg_score +
            0.20 * lexical_overlap
        )

        if score >= 0.75:
            label = "strong"
        elif score >= 0.45:
            label = "medium"
        else:
            label = "weak"

        return {
            "score": round(min(score, 1.0), 3),
            "label": label,
            "reason": "Based on rerank quality and lexical match.",
        }

    def _compute_diversity_confidence(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measures how diverse and non-redundant the retrieval result is.
        This should answer: 'Is the evidence varied enough?'
        """
        duplicate_rate = diagnostics.get("duplicate_stats", {}).get("duplicate_rate", 0.0)
        source_diversity = diagnostics.get("source_diversity", {}).get("reranked", {}).get("dominant_ratio", 1.0)

        # Lower duplicate rate and lower dominant ratio are better
        diversity_score = 1.0 - (0.6 * duplicate_rate + 0.4 * source_diversity)

        if diversity_score >= 0.75:
            label = "strong"
        elif diversity_score >= 0.45:
            label = "medium"
        else:
            label = "weak"

        return {
            "score": round(max(min(diversity_score, 1.0), 0.0), 3),
            "label": label,
            "reason": "Based on duplicate rate and source concentration.",
        }

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


    # Retrieval selection strategy - not every query needs to go through the same retrieval process
    # Some queries can be semantically stronger, some maybe keyword heavy and the rest may be more ambigous requiring metadata and/or hybrid retrieval
    def _select_retrieval_strategy(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
    ) -> Dict[str, Any]:
        """
        Pick the retrieval strategy for this query.

        Returns a dict like:
        {
            "strategy": "dense" | "sparse" | "hybrid" | "metadata_hybrid",
            "reason": "...",
            "signals": {...},
            "params": {...}
        }
        """
        q = (query or "").strip()
        q_lower = q.lower()
        tokens = re.findall(r"\b[a-z0-9_./:-]+\b", q_lower)
        token_count = len(tokens)

        metadata_cues = (
            "latest", "recent", "today", "uploaded", "upload", "file",
            "document", "doc", "source", "section", "page", "chapter",
            "appendix", "version", "pdf", "notes", "proposal", "thesis",
            "date", "when", "find"
        )

        semantic_cues = (
            "what is", "what are", "how", "why", "explain", "compare",
            "difference", "summarize", "overview", "relationship", "meaning",
            "describe", "works", "work", "concept", "idea", "insight", "implication", "consequence",
            "benefit", "advantage", "disadvantage", "pro", "con", "strength", "weakness",
            "cause", "effect", "impact", "influence", "correlation", "association",
        )

        has_metadata_cue = any(term in q_lower for term in metadata_cues)
        has_semantic_cue = any(term in q_lower for term in semantic_cues)

        # Strong lexical / exact-match signals
        has_code_like = bool(
            re.search(r"(`[^`]+`|[A-Za-z_][A-Za-z0-9_]*\(|::|->|==|!=|<=|>=)", q)
        )
        has_identifier = (
            bool(re.search(r"\b[A-Z]{2,}(?:[-_/][A-Z0-9]+)*\b", q))
            or bool(re.search(r"\b[\w.-]*\d[\w.-]*\b", q))
            or "_" in q
            or "/" in q
        )

        sparse_score = 0
        if has_code_like:
            sparse_score += 2
        if has_identifier:
            sparse_score += 1
        if len(tokens) <= 4 and not has_semantic_cue:
            sparse_score += 1

        dense_score = 0
        if has_semantic_cue:
            dense_score += 2
        if token_count >= 6:
            dense_score += 1
        if any(word in q_lower for word in ("explain", "why", "how", "difference", "compare")):
            dense_score += 1

        metadata_score = 0
        if has_metadata_cue:
            metadata_score += 2
        if "section" in q_lower or "page" in q_lower or "source" in q_lower:
            metadata_score += 1

        # Strategy decision
        if metadata_score >= 2:
            strategy = "metadata_hybrid"
        elif sparse_score >= 2 and sparse_score >= dense_score + 1:
            strategy = "sparse"
        elif dense_score >= 2 and dense_score >= sparse_score + 1:
            strategy = "dense"
        elif sparse_score > 0 and dense_score > 0:
            strategy = "hybrid"
        elif has_semantic_cue:
            strategy = "dense"
        elif has_identifier or has_code_like:
            strategy = "sparse"
        else:
            strategy = "hybrid"

        params = {
            "num_queries": num_queries,
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "adaptive_weights": adaptive_weights,
            "min_dense_similarity": min_dense_similarity,
            "min_bm25_score": min_bm25_score,
            "max_per_source": max_per_source,
        }

        if strategy == "sparse":
            params.update({
                "num_queries": min(3, max(2, num_queries - 1)),
                "retrieve_k": max(30, int(retrieve_k * 0.85)),
                "final_k": max(8, min(final_k, 12)),
                "dense_weight": 0.25,
                "sparse_weight": 0.75,
                "adaptive_weights": True,
            })
            reason = "Exact terms, identifiers, or code-like patterns look important."
        elif strategy == "dense":
            params.update({
                "num_queries": max(num_queries, 4),
                "retrieve_k": max(retrieve_k, int(retrieve_k * 1.10)),
                "final_k": max(final_k, 12),
                "dense_weight": 0.75,
                "sparse_weight": 0.25,
                "adaptive_weights": True,
                "min_dense_similarity": max(min_dense_similarity, 0.20),
            })
            reason = "This looks like a conceptual or paraphrased query."
        elif strategy == "metadata_hybrid":
            params.update({
                "num_queries": max(3, num_queries),
                "retrieve_k": max(retrieve_k, int(retrieve_k * 0.95)),
                "final_k": max(final_k, 10),
                "dense_weight": 0.5,
                "sparse_weight": 0.5,
                "adaptive_weights": True,
                "max_per_source": 2 if max_per_source is not None else None,
            })
            reason = "The query contains metadata-like constraints such as source, section, or date."
        else:
            params.update({
                "num_queries": max(3, num_queries),
                "retrieve_k": retrieve_k,
                "final_k": final_k,
                "dense_weight": 0.5,
                "sparse_weight": 0.5,
                "adaptive_weights": True if adaptive_weights else adaptive_weights,
            })
            reason = "Mixed signals. Hybrid retrieval is the safest default."

        return {
            "strategy": strategy,
            "reason": reason,
            "signals": {
                "has_metadata_cue": has_metadata_cue,
                "has_semantic_cue": has_semantic_cue,
                "has_code_like": has_code_like,
                "has_identifier": has_identifier,
                "token_count": token_count,
                "sparse_score": sparse_score,
                "dense_score": dense_score,
                "metadata_score": metadata_score,
            },
            "params": params,
        }

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
        strategy_info = self._select_retrieval_strategy(
            query=query,
            num_queries=num_queries,
            retrieve_k=retrieve_k,
            final_k=final_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=max_per_source,
        )

        p = strategy_info["params"]

        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=p["num_queries"],
        )

        query_results = []

        for q in expanded_queries:
            results = self.retriever.hybrid_search(
                q,
                retrieve_k=p["retrieve_k"],
                final_k=p["final_k"],
                k=k,
                dense_weight=p["dense_weight"],
                sparse_weight=p["sparse_weight"],
                adaptive_weights=p["adaptive_weights"],
                min_dense_similarity=p["min_dense_similarity"],
                min_bm25_score=p["min_bm25_score"],
                max_per_source=p["max_per_source"],
            )
            query_results.append(results)

        merged = self._aggregate_results(
            query_results,
            k=k,
            max_per_source=p["max_per_source"],
        )

        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=merged,
            top_k=p["final_k"],
        )

        for chunk in reranked:
            chunk["retrieval_strategy"] = strategy_info["strategy"]
            chunk["retrieval_strategy_reason"] = strategy_info["reason"]

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

        strategy_info = self._select_retrieval_strategy(
            query=query,
            num_queries=num_queries,
            retrieve_k=retrieve_k,
            final_k=final_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=max_per_source,
        )

        strategy_params = strategy_info["params"]

        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=strategy_params["num_queries"],
        )

        per_query_results = []

        for expanded_query in expanded_queries:
            results = self.retriever.hybrid_search(
                expanded_query,
                retrieve_k=strategy_params["retrieve_k"],
                final_k=strategy_params["final_k"],
                k=k,
                dense_weight=strategy_params["dense_weight"],
                sparse_weight=strategy_params["sparse_weight"],
                adaptive_weights=strategy_params["adaptive_weights"],
                min_dense_similarity=strategy_params["min_dense_similarity"],
                min_bm25_score=strategy_params["min_bm25_score"],
                max_per_source=strategy_params["max_per_source"],
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

        def _tokenize(text: str):
            STOPWORDS = {
                "a", "an", "the", "is", "are", "was", "were",
                "what", "how", "why", "when", "where",
                "and", "or", "but", "if", "then",
                "of", "in", "on", "to", "for", "with",
                "do", "does", "did", "they", "them", "their",
                "this", "that", "these", "those", "the", "like", " ",
                "it", "its", "as", "at", "by",
                "from", "about", "into", "through",
                "can", "could", "would", "should",
                "be", "been", "being",
                "vs", "versus"
            }

            tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
            filtered_tokens = [token for token in tokens if token not in STOPWORDS]
            return set(filtered_tokens)

        def _compute_lexical_overlap(query_text: str, chunks, top_n: int = 5):
            """
            Measures how much of the query vocabulary appears in retrieved chunks.

            overlap = shared_query_tokens / total_query_tokens
            """
            query_tokens = _tokenize(query_text)

            if not query_tokens or not chunks:
                return {
                    "average_lexical_overlap": 0.0,
                    "best_lexical_overlap": 0.0,
                    "per_chunk": [],
                }

            per_chunk = []

            for rank, chunk in enumerate(chunks[:top_n], start=1):
                chunk_text = chunk.get("text", "")
                chunk_tokens = _tokenize(chunk_text)

                if not chunk_tokens:
                    overlap = 0.0
                else:
                    shared = query_tokens & chunk_tokens
                    overlap = len(shared) / len(query_tokens)

                per_chunk.append(
                    {
                        "rank": rank,
                        "source": _chunk_source(chunk),
                        "overlap": round(overlap, 4),
                        "shared_terms": sorted(list(query_tokens & chunk_tokens))[:10],
                        "query_token_count": len(query_tokens),
                        "chunk_token_count": len(chunk_tokens),
                    }
                )

            avg_overlap = sum(item["overlap"] for item in per_chunk) / len(per_chunk)
            best_overlap = max(item["overlap"] for item in per_chunk)

            return {
                "average_lexical_overlap": round(avg_overlap, 4),
                "best_lexical_overlap": round(best_overlap, 4),
                "per_chunk": per_chunk,
            }

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
                # Normalize to a hashable representation (frozenset of tokens)
                token_set = _normalize_text(chunk["text"])
                key = frozenset(token_set)

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

        # Before, reranked could still leave repeated sources in the final list but now after reranking final list will be trimmed so each source appears at most max_per_source times
        reranked = self._apply_source_cap(reranked, max_per_source)

        lexical_overlap_stats = _compute_lexical_overlap(
            query_text=query,
            chunks=reranked,
            top_n=context_top_k,
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

        rerank_scores = [c.get("rerank_score", 0.0) for c in reranked]
        source_div_context = source_diversity["context"]
        merged_div_context = source_diversity["merged"]

        retrieval_health_input = {
            "rerank_scores": rerank_scores,
            "duplicate_rate": duplicate_stats["duplicate_rate"],
            "source_diversity": merged_div_context["dominant_ratio"],
            "dominant_source_ratio": merged_div_context["dominant_ratio"],
            "top3_source_concentration": merged_div_context["top_k_ratio"],
            "avg_lexical_overlap": lexical_overlap_stats["average_lexical_overlap"],
        }

        retrieval_health = self._compute_retrieval_health(retrieval_health_input)

        relevance_confidence = self._compute_relevance_confidence(retrieval_health_input)
        diversity_confidence = self._compute_diversity_confidence({
            "duplicate_stats": duplicate_stats,
            "source_diversity": source_diversity,
        })

        diagnostics = {
            "query": query,
            "expanded_queries": expanded_queries,
            "num_expanded_queries": len(expanded_queries),
            "retrieval_strategy": strategy_info["strategy"],
            "retrieval_strategy_details": strategy_info,
            "duplicate_stats": duplicate_stats,
            "rerank_lift": rerank_lift,
            "metadata_coverage": metadata_coverage,
            "source_diversity": source_diversity,
            "retrieval_health": retrieval_health,
            "lexical_overlap": lexical_overlap_stats,
            "relevance_confidence": relevance_confidence,
            "diversity_confidence": diversity_confidence,
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

    def _choose_retry_policy(
        self,
        health: Dict[str, Any],
        num_queries: int,
        retrieve_k: int,
        final_k: int,
        dense_weight: float,
        sparse_weight: float,
        max_per_source: Optional[int],
    ) -> Dict[str, Any]:
        """
        Map retrieval-health reasons to a retry strategy.
        """
        reasons = [r.lower() for r in health.get("reasons", [])]
        reasons_text = " | ".join(reasons)
        decision_trace=[]

        # Start with the current settings
        policy = {
            "num_queries": num_queries,
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "max_per_source": max_per_source,
        }

        # 1) Weak lexical overlap -> favor sparse retrieval a bit more
        if "lexical overlap is weak" in reasons_text:
            policy["num_queries"] = min(num_queries + 1, 6)
            policy["retrieve_k"] = max(policy["retrieve_k"], int(retrieve_k * 1.25))
            policy["final_k"] = max(policy["final_k"], int(final_k * 1.25))
            policy["dense_weight"] = 0.35
            policy["sparse_weight"] = 0.65
            decision_trace.append(
                {
                    "reason": "Weak lexical overlap",
                    "actions": [
                        "Boosted sparse retrieval weighting",
                        "Expanded retrieve_k",
                        "Expanded final_k",
                        "Increased query expansion",
                    ],
                }
            )
        # 2) High duplicate rate -> diversify more
        if "duplicate rate is high" in reasons_text:
            if policy["max_per_source"] is not None:
                policy["max_per_source"] = max(1, policy["max_per_source"] - 1)

            policy["retrieve_k"] = max(policy["retrieve_k"], int(retrieve_k * 1.25))
            decision_trace.append(
                {
                    "reason": "High duplicate rate",
                    "actions": [
                        "Reduced max_per_source",
                        "Expanded retrieve_k",
                    ],
                }
            )
            
        # 3) Low source diversity -> broaden search
        if "source diversity is low" in reasons_text:
            policy["num_queries"] = min(policy["num_queries"] + 1, 6)
            policy["retrieve_k"] = max(policy["retrieve_k"], int(retrieve_k * 1.40))
            policy["final_k"] = max(policy["final_k"], int(final_k * 1.20))
            # Force stronger source diversity balancing on retry if not already at 1
            policy["max_per_source"] = 1
            decision_trace.append(
                {
                    "reason": "Low source diversity",
                    "actions": [
                        "Expanded query generation",
                        "Broadened retrieval pool",
                    ],
                }
            )
        # 4) Low rerank quality -> search wider
        if (
            "average rerank score is low" in reasons_text
            or "top rerank score is weak" in reasons_text
        ):
            policy["num_queries"] = min(policy["num_queries"] + 1, 6)
            policy["retrieve_k"] = max(policy["retrieve_k"], int(retrieve_k * 1.50))
            policy["final_k"] = max(policy["final_k"], int(final_k * 1.50))
            decision_trace.append(
                {
                    "reason": "Weak rerank quality",
                    "actions": [
                        "Aggressively expanded retrieve_k",
                        "Expanded final_k",
                        "Increased query expansion",
                    ],
                }
            )

        return {
            "policy": policy,
            "decision_trace": decision_trace,
        }

    # Build a mapping from retrieval health severity to context construction parameters like how many chunks to include in the final context and how many top reranked chunks to consider when building the context.
    def _choose_context_sizes(self, health: Dict[str, Any]) -> Dict[str, int]:
        severity = health.get("severity", "medium")

        if severity == "strong":
            return {"final_k": 5, "context_top_k": 3}
        elif severity == "medium":
            return {"final_k": 10, "context_top_k": 5}
        else:
            return {"final_k": 15, "context_top_k": 7}

    # Confidence routing based on retrieval health - whether to trust retrieval as is, to be cautious or to consider retrying or abstaining.
    def _choose_confidence_route(self, diagnostics: Dict[str, Any]) -> Dict[str, str]:
        """
        Route based on relevance confidence, with domain-gate support.
        """
        relevance = diagnostics.get("relevance_confidence", {})
        diversity = diagnostics.get("diversity_confidence", {})
        domain_gate = diagnostics.get("domain_gate", {})

        relevance_label = relevance.get("label", "medium")
        diversity_label = diversity.get("label", "medium")

        in_domain = not domain_gate.get("is_ood", False)
        domain_fit = domain_gate.get("combined_fit", 0.0)

        # Strong relevance
        if relevance_label == "strong":
            if diversity_label == "weak":
                return {
                    "confidence": "medium",
                    "action": "generate_cautiously",
                    "reason": "Relevant evidence found, but it is too repetitive.",
                }
            return {
                "confidence": "high",
                "action": "generate",
                "reason": "Relevant evidence is strong enough to answer normally.",
            }

        # Medium relevance
        if relevance_label == "medium":
            return {
                "confidence": "medium",
                "action": "generate_cautiously",
                "reason": "Evidence is usable but not ideal.",
            }

        # Weak relevance, but domain gate says the query is still in-domain
        if in_domain and domain_fit >= 0.55:
            return {
                "confidence": "medium",
                "action": "generate_cautiously",
                "reason": "Query appears in-domain and probe evidence is plausible, but rerank confidence is low.",
            }

        return {
            "confidence": "low",
            "action": "retry_or_abstain",
            "reason": "Relevant evidence is too weak to trust.",
        }



    
    # A soft gate to detect out-of-domain queries before doing heavy retrieval work.
    # Runs a quick probe search against the corpus and checks if the query seems to belong to the same domain based on retrieval signals and lexical overlap.
    # This is intentionally designed to be soft and corpus-aware without hardcoded rules, relying instead on evidence from the actual index
    # If best score and overlap are too weak then it marks the query as out of domain (OOD)
    # Returns early with abstention
    def _domain_gate_preflight(
        self,
        query,
        probe_k: int = 5,
        overlap_threshold: float = 0.08,
    ):
        """
        Lightweight pre-retrieval gate.
        """
        try:
            probe_results = self.retriever.hybrid_search(
                query,
                retrieve_k=probe_k,
                final_k=min(5, probe_k),
                k=60,
                dense_weight=0.5,
                sparse_weight=0.5,
                adaptive_weights=False,
                min_dense_similarity=0.0,
                min_bm25_score=0.0,
                max_per_source=1,
            )
        except Exception as exc:
            return {
                "is_ood": True,
                "confidence": "low",
                "action": "abstain",
                "reason": f"Domain gate probe failed: {exc}",
                "best_score": 0.0,
                "best_overlap": 0.0,
                "combined_fit": 0.0,
                "probe_count": 0,
                "top_source": None,
                "top_preview": "",
            }

        STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were",
            "what", "how", "why", "when", "where", "which",
            "and", "or", "but", "if", "then",
            "of", "in", "on", "to", "for", "with",
            "do", "does", "did", "they", "them", "their",
            "this", "that", "these", "those", "like",
            "it", "its", "as", "at", "by",
            "from", "about", "into", "through",
            "can", "could", "would", "should",
            "be", "been", "being",
            "vs", "versus"
        }

        q_tokens = set(re.findall(r"\b[a-z0-9]+\b", (query or "").lower()))
        q_tokens = {t for t in q_tokens if t not in STOPWORDS}
        if not q_tokens:
            q_tokens = set(re.findall(r"\b[a-z0-9]+\b", (query or "").lower()))

        best_score = 0.0
        best_overlap = 0.0
        best_source = None
        best_preview = ""

        for chunk in probe_results[:probe_k]:
            text = chunk.get("text", "")
            chunk_tokens = set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
            chunk_tokens = {t for t in chunk_tokens if t not in STOPWORDS}

            overlap = (len(q_tokens & chunk_tokens) / len(q_tokens)) if q_tokens else 0.0

            score = max(
                float(chunk.get("cosine_score", 0.0) or 0.0),
                float(chunk.get("rerank_score", 0.0) or 0.0),
                float(chunk.get("rrf_score", 0.0) or 0.0),
                float(chunk.get("multi_query_rrf_score", 0.0) or 0.0),
            )

            if score > best_score:
                best_score = score
                best_overlap = overlap
                best_source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
                best_preview = text[:140].replace("\n", " ")

        combined_fit = 0.7 * best_score + 0.3 * best_overlap

        if combined_fit >= 0.55:
            return {
                "is_ood": False,
                "confidence": "high" if combined_fit >= 0.70 else "medium",
                "action": "continue",
                "reason": (
                    f"Probe retrieval looks plausible "
                    f"(combined_fit={combined_fit:.3f}, best_score={best_score:.3f}, "
                    f"best_overlap={best_overlap:.3f})."
                ),
                "best_score": round(best_score, 4),
                "best_overlap": round(best_overlap, 4),
                "combined_fit": round(combined_fit, 4),
                "probe_count": len(probe_results),
                "top_source": best_source,
                "top_preview": best_preview,
            }

        return {
            "is_ood": True,
            "confidence": "low",
            "action": "abstain",
            "reason": (
                f"Probe retrieval is too weak "
                f"(combined_fit={combined_fit:.3f}, best_score={best_score:.3f}, "
                f"best_overlap={best_overlap:.3f}); query may be out of domain."
            ),
            "best_score": round(best_score, 4),
            "best_overlap": round(best_overlap, 4),
            "combined_fit": round(combined_fit, 4),
            "probe_count": len(probe_results),
            "top_source": best_source,
            "top_preview": best_preview,
        }

    # Adaptive retrieval controller with retry logic based on diagnostics
    def adaptive_search_with_retry(
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
        Adaptive retrieval controller.

        Step 1:
            Run normal retrieval + diagnostics.

        Step 2:
            Inspect retrieval health.

        Step 3:
            If retrieval is weak, retry with a reason-aware strategy.

        Returns:
            {
                "used_retry": bool,
                "retry_params": {...} | None,
                "initial_result": ...,
                "final_result": ...
            }
        """

        # This is the actual gate check before retrieving
        # If query looks OOD then no expansion, no retrieval, no reranking and no retry loop
        domain_gate = self._domain_gate_preflight(query)

        if domain_gate and domain_gate["is_ood"]:
            empty_diag = {
                "query": query,
                "domain_gate": domain_gate,
                "retrieval_health": {
                    "is_weak": True,
                    "severity": "weak",
                    "risk_score": 1.0,
                    "reasons": [domain_gate["reason"]],
                    "summary": {},
                },
                "relevance_confidence": {
                    "score": 0.0,
                    "label": "weak",
                    "reason": "Blocked by domain gate before retrieval.",
                },
                "diversity_confidence": {
                    "score": 0.0,
                    "label": "weak",
                    "reason": "No in-domain evidence retrieved.",
                },
            }

            empty_result = {
                "query": query,
                "expanded_queries": [],
                "per_query_results": [],
                "merged_chunks": [],
                "reranked_chunks": [],
                "context_chunks": [],
                "context": "",
                "diagnostics": empty_diag,
            }

            return {
                "used_retry": False,
                "confidence_route": {
                    "confidence": "low",
                    "action": "retry_or_abstain",
                    "reason": "Domain gate detected an out-of-domain query.",
                },
                "decision_trace": [
                    {
                        "reason": "Domain gate blocked the query",
                        "actions": [
                            "Skipped retrieval",
                            "Skipped reranking",
                            "Abstained early",
                        ],
                    }
                ],
                "chosen_final_k": 0,
                "chosen_context_top_k": 0,
                "retry_params": None,
                "initial_result": empty_result,
                "final_result": empty_result,
            }

        initial_result = self.search_with_diagnostics(
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
            context_top_k=context_top_k,
            include_scores=include_scores,
        )

        # Attach gate info to the initial result so it always shows up in diagnostics
        initial_result["diagnostics"]["domain_gate"] = domain_gate

        initial_health = initial_result["diagnostics"].get("retrieval_health", {})
        is_weak = initial_health.get("is_weak", False)
        initial_route = self._choose_confidence_route(initial_result["diagnostics"])

        # System does not always need the same amount of context as too little context misses evidence
        # too much context adds noise
        # the right amount depends on retrieval quality
        # added dynamic context sizing which chooses how many chunks to include in the final context 
        # and how many top reranked chunks to consider when building the context based on the initial retrieval health assessment
        size_policy = self._choose_context_sizes(initial_health)
        chosen_final_k = size_policy["final_k"]
        chosen_context_top_k = size_policy["context_top_k"]

        # If not weak, no retry.
        if not is_weak:
            return {
                "chosen_final_k": chosen_final_k,
                "chosen_context_top_k": chosen_context_top_k,
                "used_retry": False,
                "confidence_route": initial_route,
                "retry_params": None,
                "initial_result": initial_result,
                "final_result": initial_result,
            }

        # Weak retrieval -- choose a retry policy based on the reasons
        retry_decision = self._choose_retry_policy(
            health=initial_health,
            num_queries=num_queries,
            retrieve_k=retrieve_k,
            final_k=final_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            max_per_source=max_per_source,
        )

        retry_policy = retry_decision["policy"]
        decision_trace = retry_decision["decision_trace"]

        retry_result = self.search_with_diagnostics(
            query=query,
            num_queries=retry_policy["num_queries"],
            retrieve_k=retry_policy["retrieve_k"],
            final_k=chosen_final_k,
            k=k,
            dense_weight=retry_policy["dense_weight"],
            sparse_weight=retry_policy["sparse_weight"],
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=retry_policy["max_per_source"],
            context_top_k=chosen_context_top_k,
            include_scores=include_scores,
        )

        retry_result["diagnostics"]["domain_gate"] = domain_gate
        final_route = self._choose_confidence_route(retry_result["diagnostics"])

        return {
            "used_retry": True,
            "confidence_route": final_route,
            "decision_trace": decision_trace,
            "chosen_final_k": chosen_final_k,
            "chosen_context_top_k": chosen_context_top_k,
            "retry_params": {
                "num_queries": retry_policy["num_queries"],
                "retrieve_k": retry_policy["retrieve_k"],
                "final_k": chosen_final_k,
                "dense_weight": retry_policy["dense_weight"],
                "sparse_weight": retry_policy["sparse_weight"],
                "max_per_source": retry_policy["max_per_source"],
            },
            "initial_result": initial_result,
            "final_result": retry_result,
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

        print("\nRETRIEVAL STRATEGY:")

        strategy = diag.get("retrieval_strategy", "unknown")
        strategy_details = diag.get("retrieval_strategy_details", {})

        print(f"  Strategy: {strategy}")
        print(f"  Reason: {strategy_details.get('reason', '')}")

        print("\nDUPLICATE STATS:")
        dup = diag["duplicate_stats"]
        print(f"  Total candidates: {dup['total_candidates']}")
        print(f"  Unique candidates: {dup['unique_candidates']}")
        print(f"  Duplicate candidates: {dup['duplicate_candidates']}")
        print(f"  Duplicate rate: {dup['duplicate_rate']:.4f}")

        # print("\nRERANK LIFT:")
        # lift = diag["rerank_lift"]
        # print(f"  Compared candidates: {lift['compared_candidates']}")
        # print(f"  Average lift: {lift['average_lift']:.2f}")
        # print(f"  Average absolute lift: {lift['average_abs_lift']:.2f}")
        # print(f"  Promoted: {lift['promoted']}")
        # print(f"  Demoted: {lift['demoted']}")
        # print(f"  Unchanged: {lift['unchanged']}")

        # print("\nTOP GAINS:")
        # if lift["top_gains"]:
        #     for idx, item in enumerate(lift["top_gains"], start=1):
        #         print(f"\n[GAIN {idx}]")
        #         print(f"Source: {item['source']}")
        #         print(f"Before Rank: {item['before_rank']}")
        #         print(f"After Rank: {item['after_rank']}")
        #         print(f"Lift: +{item['lift']}")
        #         print(f"Rerank Score: {item['rerank_score']:.4f}")
        #         print(f"RRF Score: {item['rrf_score']:.6f}")
        #         print("Preview:")
        #         print(item["text_preview"])
        #         print("-" * 60)
        # else:
        #     print("None")

        # print("\nTOP LOSSES:")
        # if lift["top_losses"]:
        #     for idx, item in enumerate(lift["top_losses"], start=1):
        #         print(f"\n[LOSS {idx}]")
        #         print(f"Source: {item['source']}")
        #         print(f"Before Rank: {item['before_rank']}")
        #         print(f"After Rank: {item['after_rank']}")
        #         print(f"Lift: {item['lift']}")
        #         print(f"Rerank Score: {item['rerank_score']:.4f}")
        #         print(f"RRF Score: {item['rrf_score']:.6f}")
        #         print("Preview:")
        #         print(item["text_preview"])
        #         print("-" * 60)
        # else:
        #     print("None")

        print("\nRETRIEVAL HEALTH:")
        health = diag.get("retrieval_health", {})
        print(f"  Severity: {health.get('severity', 'unknown')}")
        print(f"  Risk score: {health.get('risk_score', 0.0)}")
        print(f"  Is weak: {health.get('is_weak', False)}")

        reasons = health.get("reasons", [])
        if reasons:
            print("  Reasons:")
            for reason in reasons:
                print(f"    - {reason}")
        else:
            print("  Reasons: none")

        print("\nMETADATA COVERAGE:")

        for stage_name, stage_cov in diag["metadata_coverage"].items():
            print(f"\n{stage_name.upper()} CHUNKS:")
            for field, stats in stage_cov.items():
                print(
                    f"  {field}: {stats['present']}/{stats['total']} "
                    f"({stats['coverage_rate']:.4f})"
                )

        print("\nLEXICAL OVERLAP:")

        lex = diag.get("lexical_overlap", {})
        print(f"  Average lexical overlap: {lex.get('average_lexical_overlap', 0.0):.4f}")
        print(f"  Best lexical overlap: {lex.get('best_lexical_overlap', 0.0):.4f}")

        per_chunk = lex.get("per_chunk", [])
        if per_chunk:
            print("  Per chunk:")
            for item in per_chunk:
                print(
                    f"    Chunk {item['rank']} | "
                    f"Source: {item['source']} | "
                    f"Overlap: {item['overlap']:.4f} | "
                    f"Shared terms: {item['shared_terms']}"
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

        # print("\n==================================================")
        # print("PER-QUERY RETRIEVAL")
        # print("==================================================")

        # for i, item in enumerate(diag["per_query_results"], start=1):
        #     print(f"\nQUERY VARIANT [{i}]")
        #     print(f"Query: {item['query']}")
        #     print(f"Retrieved Count: {item['count']}")

        #     print("\nSources:")
        #     for s in item["sources"]:
        #         print(f"  - {s}")

        #     print("\nSource Distribution:")
        #     for source, count in item["source_distribution"].items():
        #         print(f"  {source}: {count}")

        #     print(f"\nTop Cosine Scores: {item['top_cosine_scores']}")
        #     print(f"Top RRF Scores: {item['top_rrf_scores']}")

        # print("\n==================================================")
        # print("MERGED RESULTS")
        # print("==================================================")

        # merged = diag["merged_results"]

        # print(f"Merged Candidate Count: {merged['count']}")

        # print("\nMerged Source Distribution:")
        # for source, count in merged["source_distribution"].items():
        #     print(f"  {source}: {count}")

        # print(f"\nTop Multi-Query RRF Scores:")
        # print(merged["top_multi_query_rrf_scores"])

        # print("\n==================================================")
        # print("RERANKED RESULTS")
        # print("==================================================")

        # reranked = diag["reranked_results"]

        # print(f"Reranked Count: {reranked['count']}")

        # print("\nReranked Source Distribution:")
        # for source, count in reranked["source_distribution"].items():
        #     print(f"  {source}: {count}")

        # print(f"\nTop Rerank Scores:")
        # print(reranked["top_rerank_scores"])

        # print("\n==================================================")
        # print("FINAL CONTEXT SUMMARY")
        # print("==================================================")

        # context = diag["context_summary"]

        # print(f"Context Chunk Count: {context['count']}")

        # print("\nContext Sources:")
        # for source, count in context["source_distribution"].items():
        #     print(f"  {source}: {count}")


    def print_adaptive_result(self, adaptive_result):
        """
        Pretty-print adaptive retrieval results.
        """

        initial_health = adaptive_result["initial_result"]["diagnostics"].get(
            "retrieval_health", {}
        )
        final_health = adaptive_result["final_result"]["diagnostics"].get(
            "retrieval_health", {}
        )


        print("\n==================================================")
        print("DYNAMIC CONTEXT SIZING POLICY")
        print("==================================================")

        print(f"  chosen_final_k: {adaptive_result.get('chosen_final_k')}")
        print(f"  chosen_context_top_k: {adaptive_result.get('chosen_context_top_k')}")

        relevance = adaptive_result["final_result"]["diagnostics"].get("relevance_confidence", {})
        diversity = adaptive_result["final_result"]["diagnostics"].get("diversity_confidence", {})

        strategy = adaptive_result["final_result"]["diagnostics"].get(
            "retrieval_strategy", "unknown"
        )

        strategy_details = adaptive_result["final_result"]["diagnostics"].get(
            "retrieval_strategy_details", {}
        )

        domain_gate = adaptive_result["final_result"]["diagnostics"].get("domain_gate", {})

        print("\n==================================================")
        print("DOMAIN GATE / OOD DETECTION")
        print("==================================================")

        if domain_gate:
            print(f"  is_ood: {domain_gate.get('is_ood', False)}")
            print(f"  confidence: {domain_gate.get('confidence', 'unknown')}")
            print(f"  action: {domain_gate.get('action', 'unknown')}")
            print(f"  reason: {domain_gate.get('reason', '')}")
            print(f"  best_score: {domain_gate.get('best_score', 0.0)}")
            print(f"  best_overlap: {domain_gate.get('best_overlap', 0.0)}")
            print(f"  combined_fit: {domain_gate.get('combined_fit', 0.0)}")
            print(f"  probe_count: {domain_gate.get('probe_count', 0)}")
            print(f"  top_source: {domain_gate.get('top_source', 'unknown')}")
            print(f"  top_preview: {domain_gate.get('top_preview', '')}")
        else:
            print("  domain gate: not run")

        print("\n==================================================")
        print("RETRIEVAL STRATEGY SELECTION")
        print("==================================================")

        print(f"  Strategy: {strategy}")
        print(f"  Reason: {strategy_details.get('reason', '')}")

        signals = strategy_details.get("signals", {})
        if signals:
            print("\n  Signals:")
            for key, value in signals.items():
                print(f"    {key}: {value}")

        params = strategy_details.get("params", {})
        if params:
            print("\n  Selected Parameters:")
            for key, value in params.items():
                print(f"    {key}: {value}")

        print("\n==================================================")
        print("\nRELEVANCE CONFIDENCE")
        print("\n==================================================")
        print(f"  score: {relevance.get('score', 0.0)}")
        print(f"  label: {relevance.get('label', 'unknown')}")
        print(f"  reason: {relevance.get('reason', '')}")

        print("\nDIVERSITY CONFIDENCE")
        print(f"  score: {diversity.get('score', 0.0)}")
        print(f"  label: {diversity.get('label', 'unknown')}")
        print(f"  reason: {diversity.get('reason', '')}")

        confidence_route = adaptive_result.get("confidence_route", {})

        print("\n==================================================")
        print("\nCONFIDENCE ROUTE")
        print("\n==================================================")
        print(f"  confidence: {confidence_route.get('confidence', 'unknown')}")
        print(f"  action: {confidence_route.get('action', 'unknown')}")
        print(f"  reason: {confidence_route.get('reason', '')}")

        print("\n==================================================")
        print("ADAPTIVE RETRIEVAL")
        print("==================================================")
        print(f"Used retry: {adaptive_result['used_retry']}")

        if adaptive_result["retry_params"] is not None:
            print("Retry params:")
            for key, value in adaptive_result["retry_params"].items():
                print(f"  {key}: {value}")

            decision_trace = adaptive_result.get("decision_trace", [])

            if decision_trace:
                print("\nRETRY DECISION TRACE")

                for i, step in enumerate(decision_trace, start=1):
                    print(f"\n[{i}] {step['reason']}")

                    for action in step["actions"]:
                        print(f"    -> {action}")

        print("\nINITIAL HEALTH")
        print(f"  Severity: {initial_health.get('severity', 'unknown')}")
        print(f"  Risk score: {initial_health.get('risk_score', 0.0)}")
        print(f"  Is weak: {initial_health.get('is_weak', False)}")

        reasons = initial_health.get("reasons", [])
        if reasons:
            print("  Reasons:")
            for reason in reasons:
                print(f"    - {reason}")

        print("\nFINAL HEALTH")
        print(f"  Severity: {final_health.get('severity', 'unknown')}")
        print(f"  Risk score: {final_health.get('risk_score', 0.0)}")
        print(f"  Is weak: {final_health.get('is_weak', False)}")

        reasons = final_health.get("reasons", [])
        if reasons:
            print("  Reasons:")
            for reason in reasons:
                print(f"    - {reason}")

        print("\nFINAL TOP RESULTS")
        for i, c in enumerate(adaptive_result["final_result"]["reranked_chunks"][:5], start=1):
            source = c.get("source", "unknown")
            rerank_score = c.get("rerank_score", 0.0)
            rrf_score = c.get("multi_query_rrf_score", 0.0)
            print(f"  [{i}] {source} | rerank={rerank_score:.4f} | rrf={rrf_score:.4f}")

        # print("\n==================================================")
        # print("ADAPTIVE RETRIEVAL")
        # print("==================================================")

        # print(f"Used retry: {adaptive_result['used_retry']}")

        # if adaptive_result["retry_params"] is not None:
        #     print("\nRetry params:")
        #     for key, value in adaptive_result["retry_params"].items():
        #         print(f"  {key}: {value}")

        # print("\n==================================================")
        # print("INITIAL RETRIEVAL DIAGNOSTICS")
        # print("==================================================")

        # self.print_diagnostics(
        #     adaptive_result["initial_result"]["diagnostics"]
        # )

        # print("\n==================================================")
        # print("FINAL RETRIEVAL DIAGNOSTICS")
        # print("==================================================")

        # self.print_diagnostics(
        #     adaptive_result["final_result"]["diagnostics"]
        # )

        # print("\n==================================================")
        # print("FINAL RERANKED RESULTS")
        # print("==================================================")

        # for i, c in enumerate(
        #     adaptive_result["final_result"]["reranked_chunks"],
        #     start=1,
        # ):
        #     print(f"\n[{i}] source={c.get('source', 'unknown')}")

        #     if "rerank_score" in c:
        #         print(f"rerank_score={c['rerank_score']:.4f}")

        #     if "multi_query_rrf_score" in c:
        #         print(
        #             f"multi_query_rrf_score="
        #             f"{c['multi_query_rrf_score']:.4f}"
        #         )

        #     print(c.get("text", "")[:300])
        #     print("-" * 80)

        # print("\n==================================================")
        # print("PROMPT-READY CONTEXT")
        # print("==================================================")

        # print(adaptive_result["final_result"]["context"])

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

    # test_queries = [
    #     "What is dense and sparse retrieval?",
    #     "that thing where llms forget the middle part",
    #     "how do vector databases work",
    #     "bm25 vs embeddings",
    # ]

    # For adaptive search testing, use queries that are more likely to trigger reason-aware retrieval
    test_queries = [ 
        "that issue where transformers ignore middle context", 
        "list the countries in the world", 
        "lost in the middle attention problem", 
        "fix my wifi router",
        "what is dense and sparse retrieval?" ]

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

        # result = multi_query_retriever.search_with_diagnostics(
        #     query,
        #     num_queries=4,
        #     retrieve_k=40,
        #     final_k=10,
        #     context_top_k=5,
        # )

        # For demonstration, we'll call the adaptive search which includes diagnostics and potential retry logic
        result = multi_query_retriever.adaptive_search_with_retry(
            query,
            num_queries=4,
            retrieve_k=40,
            final_k=10,
            context_top_k=5,
        )

        multi_query_retriever.print_adaptive_result(result)

        generator = Generator()

        # Use the result returned from adaptive_search_with_retry
        answer = generator.generate(
            query=query,
            context=result["final_result"]["context"],
            confidence_route=result.get("confidence_route")
        )

        # final_result = result["final_result"]


        # print("\n==================================================")
        # print("FINAL RERANKED RESULTS")
        # print("==================================================")

        # for i, c in enumerate(final_result["reranked_chunks"], start=1):
        #     print(f"\n[{i}] source={c.get('source', 'unknown')}")

        #     if "rerank_score" in c:
        #         print(f"rerank_score={c['rerank_score']:.4f}")

        #     if "multi_query_rrf_score" in c:
        #         print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")

        #     print(c.get("text", "")[:300])

        #     print("-" * 80)

        print("\n==================================================")
        print("PROMPT-READY CONTEXT")
        print("==================================================")

        print(result["final_result"]["context"])

        print("\n==================================================")
        print("FINAL LLM ANSWER")
        print("==================================================")
        print(answer)