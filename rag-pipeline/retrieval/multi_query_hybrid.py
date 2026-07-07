from typing import Dict, Any, List, Optional
import io
import sys
import time
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
from observability.metrics_logger import MetricsLogger
from caching.query_cache import QueryCache
from caching.retrieval_cache import RetrievalCache


class MultiQueryHybridRetriever:
    def __init__(self, index_path=None, model_name="qwen2.5:7b", client=None):
        self.retriever = Retriever(index_path=index_path)
        self.query_generator = MultiQueryRetriever(
            model_name=model_name,
            client=client,
        )
        self.reranker = Reranker()
        self.metrics_logger = MetricsLogger()
        self.query_cache = QueryCache()
        self.retrieval_cache = RetrievalCache()
        
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

        if score_gap < 0.03:
            risk_score += 0.10
            reasons.append(f"Rerank scores are flat (top gap {score_gap:.3f}).")

        if score_std < 0.02:
            risk_score += 0.05
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

        if lexical_overlap < 0.10:
            risk_score += 0.10
            reasons.append(
                f"Lexical overlap is weak ({lexical_overlap:.3f})."
            )

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.70:
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

    # Relevance confidence asks "Did I retrieve good chunks"
    # Confidence confidence asks "What should the pipeline do next - generate, retry or abstain?"
    # Answerability confidence asks "Given the retrieved chunks, do I have enough to answer the question?"
    # Generation routing asks "How should the answer be written - normally, cautiously, or refuse to answer?"
    # Answerability teaches the concept of evidence coverage

    def _compute_answerability(
        self,
        query: str,
        reranked_chunks: List[Dict[str, Any]],
        relevance_confidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Estimates whether retrieved evidence is sufficient
        to answer the user's query

        This is NOT retrieval quality.

        This is:
            "Can I answer the question completely?"
        """

        query_tokens = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
        )

        STOPWORDS = {
            "what", "is", "are", "the", "a", "an",
            "how", "why", "when", "where",
            "and", "or", "of", "to", "for",
            "in", "on", "with", "about",
            "compare", "difference", "between",
            "explain", "describe", "then", "do", "does", "did",
            "they", "them", "their", "this", "that", "these", "those",
            "work", "concept", "idea", "insight", "implication", "consequence",
            "tell", "show", "which", "who", "whom", "whose", "where", "when",
            "do", "does", "did", "can", "could", "would", "should",
        }

        semantic_scores = []

        for chunk in reranked_chunks[:10]:
            score = chunk.get("rerank_score", 0.0)
            semantic_scores.append(score)

        semantic_score = max(semantic_scores) if semantic_scores else 0.0

        query_tokens = {
            token
            for token in query_tokens
            if token not in STOPWORDS
        }

        SYNONYM_EXPANSIONS = {
            "invented": {
                "invented",
                "created",
                "creator",
                "developed",
                "introduced",
                "published",
                "author",
                "authors",
                "researchers",
            },
        }

        expanded_query_tokens = set()

        for token in query_tokens:
            expanded_query_tokens.add(token)

            if token in SYNONYM_EXPANSIONS:
                expanded_query_tokens.update(
                    SYNONYM_EXPANSIONS[token]
                )

        if not query_tokens:
            query_tokens = set(
                re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
            )

        combined_text = " ".join(
            chunk.get("text", "").lower()
            for chunk in reranked_chunks[:10]
        )

        covered_terms = [
            token
            for token in expanded_query_tokens
            if token in combined_text
        ]

        original_query_terms = len(query_tokens)

        coverage_score = (
            min(len(covered_terms), original_query_terms)
            / original_query_terms
            if original_query_terms
            else 0.0
        )

        relevance_score = relevance_confidence.get("score", 0.0)

        answerability_score = (
            0.30 * coverage_score +
            0.30 * relevance_score +
            0.40 * semantic_score
        )

        if answerability_score >= 0.60:
            label = "high"
            can_answer = True

        elif answerability_score >= 0.30:
            label = "medium"
            can_answer = True

        else:
            label = "low"
            can_answer = False

        return {
            "score": round(answerability_score, 3),
            "label": label,
            "can_answer": can_answer,
            "coverage_score": round(coverage_score, 3),
            "semantic_score": round(semantic_score, 3),
            "covered_terms": covered_terms,
            "query_terms": sorted(query_tokens),
            "reason": (
                "Combines query-term coverage and retrieval relevance."
            ),
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


    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Lightweight rule-based query intent classifier.

        Returns:
        {
            "intent": "...",
            "reason": "..."
        }
        """

        q = (query or "").strip().lower()

        comparison_patterns = [
            " vs ",
            " versus ",
            "difference between",
            "compare",
            "comparison",
            "contrast",
            "similarity",
        ]

        howto_patterns = [
            "how do",
            "how to",
            "steps",
            "implement",
            "build",
            "create",
            "develop",
            "fix",
            "troubleshoot",
            "guide",
            "tutorial",
            "walkthrough",
            "example",
            "demo",
        ]

        fact_lookup_patterns = [
            "who ",
            "when ",
            "where ",
            "which ",
        ]

        definition_patterns = [
            "what is",
            "what are",
            "define",
            "meaning of",
            "explain",
            "definition",
            "concept",
            "idea",
            "insight",
            "implication",
            "consequence",
        ]

        exploration_patterns = [
            "explain",
            "overview",
            "summarize",
            "tell me about",
            "describe",
            "relationship between",
        ]

        if any(p in q for p in comparison_patterns):
            return {
                "intent": "comparison",
                "reason": "Comparison language detected."
            }

        if any(p in q for p in howto_patterns):
            return {
                "intent": "how_to",
                "reason": "Procedural language detected."
            }

        if any(q.startswith(p) for p in fact_lookup_patterns):
            return {
                "intent": "fact_lookup",
                "reason": "Specific factual information requested."
            }

        if any(p in q for p in exploration_patterns):
            return {
                "intent": "exploration",
                "reason": "Broad exploratory question."
            }

        if any(p in q for p in definition_patterns):
            return {
                "intent": "definition",
                "reason": "Definition-style question."
            }

        return {
            "intent": "general",
            "reason": "No strong intent detected."
        }


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
        intent_info = self._classify_query_intent(query)
        intent = intent_info["intent"]
        tokens = re.findall(r"\b[a-z0-9_./:-]+\b", q_lower)
        token_count = len(tokens)

        metadata_cues = (
            "latest", "recent", "today", "uploaded", "upload", "file",
            "document", "doc", "source", "section", "page", "chapter",
            "appendix", "version", "pdf", "notes", "proposal", "thesis",
            "date", "find"
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
            or bool(re.search(r"\b[a-z]+[A-Z][a-zA-Z]*\b", q))  # camelCase: ReAct, LoRA, QLoRA, AutoGPT
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
        if intent == "fact_lookup":
            dense_score +=2

        metadata_score = 0

        metadata_terms_found = [
            term for term in metadata_cues
            if term in q_lower
        ]

        metadata_score += len(metadata_terms_found)
        if "section" in q_lower or "page" in q_lower or "source" in q_lower:
            metadata_score += 1

        # Strategy decision
        if metadata_score >= 3:
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

        if intent == "comparison":
            params["retrieve_k"] = max(params["retrieve_k"], 80)
            params["final_k"] = max(params["final_k"], 20)

        elif intent == "how_to":
            params["retrieve_k"] = max(params["retrieve_k"], 70)
            params["final_k"] = max(params["final_k"], 15)

        elif intent == "fact_lookup":
            params["retrieve_k"] = max(params["retrieve_k"], 50)
            params["final_k"] = max(params["final_k"], 10)
            params["dense_weight"] = 0.7
            params["sparse_weight"] = 0.3

        elif intent == "definition":
            params["retrieve_k"] = min(params["retrieve_k"], 40)
            params["final_k"] = min(params["final_k"], 10)

            if strategy == "dense":
                original_strategy = strategy
                strategy = "hybrid"
                params["dense_weight"] = 0.5
                params["sparse_weight"] = 0.5
                params["min_dense_similarity"] = min_dense_similarity   # reset from dense branch's raised 0.20 back to the caller's original value
                reason = (
                    f"Definition-intent override: scoring picked '{original_strategy}', "
                    f"but definition queries default to hybrid for keyword safety."
                )

        elif intent == "exploration":
            params["retrieve_k"] = max(params["retrieve_k"], 90)
            params["final_k"] = max(params["final_k"], 20)

        return {
            "strategy": strategy,
            "reason": reason,
            "query_intent": intent,
            "query_intent_reason": intent_info["reason"],
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
        skip_retrieval_cache=False,
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

        search_start = time.perf_counter()

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

        retrieval_cache_hit = False

        strategy_name = strategy_info["strategy"]

        # Time the query expansion step and note cache hits
        query_expansion_start = time.perf_counter()

        # Implementing a simple in-memory cache for query expansions to speed up repeated queries and analyze 
        # cache hit rates in diagnostics

        expanded_queries = []
        query_cache_hit = False

        if self.query_cache.exists(query):

            expanded_queries = self.query_cache.get(query)

            query_cache_hit = True

        else:

            expanded_queries = (
                self.query_generator.generate_queries(
                    query,
                    num_queries=strategy_params["num_queries"],
                )
            )

            self.query_cache.set(
                query,
                expanded_queries,
            )

        print(
            f"[QUERY CACHE] Hit={query_cache_hit}"
        )

        query_expansion_latency_ms = round(
            (time.perf_counter() - query_expansion_start) * 1000,
            2
        )

        # print("\nDEBUG RETRIEVAL CACHE")
        # print(f"query={query}")
        # print(f"strategy={strategy_name}")

        # exists = self.retrieval_cache.exists(
        #     query,
        #     strategy_name,
        # )

        # print(f"exists={exists}")

        # if exists:

        print("\nCACHE KEY DEBUG")
        print(f"query={query}")
        print(f"strategy={strategy_name}")
        print(f"retrieve_k={strategy_params['retrieve_k']}")
        print(f"final_k={strategy_params['final_k']}")
        print(f"num_queries={strategy_params['num_queries']}")

        # Initially we got retry success rate as 0%
        # Issue was weak retrieval detected -- retry requested -- cache hit -- same retrieval result -- same diagnostics -- no improvement
        # And this adaptive retrieval never actually ran
        # But now we are temporarily disabling retrieval cache during retries to break this loop and get some data on how well the 
        # adaptive retrieval is working and whether it is improving retrieval quality and diagnostics on retry attempts


        if (
            not skip_retrieval_cache
            and self.retrieval_cache.exists(
                query,
                strategy_name
            )
        ):
        

            cached_result = self.retrieval_cache.get(
                query,
                strategy_name,
            )

            if cached_result is not None:

                retrieval_cache_hit = True

                print("[RETRIEVAL CACHE] HIT")
                print(f"query={query}")
                print(f"strategy={strategy_name}")

                cached_result["diagnostics"]["cache"][
                    "retrieval_cache_hit"
                ] = True

                return cached_result

        per_query_results = []

        retrieval_start = time.perf_counter()
        retrieval_latency_ms = 0.0

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

            retrieval_latency_ms = round(
                (time.perf_counter() - retrieval_start) * 1000,
                2
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

        fusion_start = time.perf_counter()

        merged = self._aggregate_results(
            query_result_lists,
            k=k,
            max_per_source=max_per_source,
        )

        fusion_latency_ms = round(
            (time.perf_counter() - fusion_start) * 1000,
            2
        )

        # Keep a copy before reranking because reranker mutates chunks in place.

        merged_before_rerank = [chunk.copy() for chunk in merged]

        rerank_start = time.perf_counter()

        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=merged,
            top_k=final_k,
        )

        rerank_latency_ms = round(
            (time.perf_counter() - rerank_start) * 1000,
            2
        )

        rerank_candidates_count = len(merged)
        latency_per_candidate_ms = round(
            rerank_latency_ms / max(rerank_candidates_count, 1),
            2
        )

        # Before, reranked could still leave repeated sources in the final list but now after reranking final list will be trimmed so each source appears at most max_per_source times
        reranked = self._apply_source_cap(reranked, max_per_source)

        diagnostics_start = time.perf_counter()

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

        answerability = self._compute_answerability(
            query=query,
            reranked_chunks=reranked,
            relevance_confidence=relevance_confidence,
        )

        diagnostics_latency_ms = round(
            (time.perf_counter() - diagnostics_start) * 1000,
            2
        )

        total_search_latency_ms = round(
            (time.perf_counter() - search_start) * 1000,
            2
        )

        diagnostics = {
            "query": query,
            "expanded_queries": expanded_queries,
            "num_expanded_queries": len(expanded_queries),
            "query_intent": strategy_info.get("query_intent"),
            "query_intent_reason": strategy_info.get("query_intent_reason"),
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
            "answerability": answerability,
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
            "merged_chunks": merged_before_rerank,
            "reranked_results": {
                **_summarize_chunks(reranked, score_key="rerank_score"),
                "top_rerank_scores": [
                    round(c.get("rerank_score", 0.0), 4)
                    for c in reranked[:5]
                ],
            },
            "reranked_chunks": reranked,
            "context_summary": {
                "count": len(context_chunks),
                "sources": sorted({_chunk_source(c) for c in context_chunks}),
                "source_distribution": dict(Counter(_chunk_source(c) for c in context_chunks)),
            },

            "latency": {
                "query_expansion_latency_ms": query_expansion_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "fusion_latency_ms": fusion_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "diagnostics_latency_ms": diagnostics_latency_ms,
                "total_search_latency_ms": total_search_latency_ms,
                "rerank_candidates_count": rerank_candidates_count,
                "latency_per_candidate_ms": latency_per_candidate_ms,
            },

            "cache": {
                "query_cache_hit": query_cache_hit,
                "query_expansion_latency_ms":
                    query_expansion_latency_ms,
                "retrieval_cache_hit": retrieval_cache_hit,
            }
        }


        result = {
            "query": query,
            "expanded_queries": expanded_queries,
            "per_query_results": per_query_results,
            "merged_chunks": merged_before_rerank,
            "reranked_chunks": reranked,
            "context_chunks": context_chunks,
            "context": context,
            "diagnostics": diagnostics,
        }

        # print(f"strategy_name={strategy_name}")

        self.retrieval_cache.set(
            query,
            strategy_name,
            result,
        )

        return result

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
        answerability = diagnostics.get("answerability", {})
        domain_gate = diagnostics.get("domain_gate", {})

        relevance_label = relevance.get("label", "medium")
        diversity_label = diversity.get("label", "medium")

        in_domain = not domain_gate.get("is_ood", False)
        domain_fit = domain_gate.get("combined_fit", 0.0)

        # Answerability failed and domain gate thinks it's OOD -> low confidence and retry/abstain
        # Answerability failure alone is not a death sentence if domain gate thinks it's in-domain
        # and has some fit, because maybe retrieval just missed the mark but the
        # query is still valid for the corpus. But if domain gate also thinks it is OOD then
        # it is a stronger signal that the query might be fundamentally mismatched
        # with the corpus.
        # So, answerability failure + query in domain then let LLM try


        if (
            not answerability.get("can_answer", True)
            and domain_gate.get("is_ood", False)
        ):
            return {
                "confidence": "low",
                "action": "retry_or_abstain",
                "reason": "Retrieved evidence does not sufficiently cover the query.",
            }

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

        # If answerability says we can answer, let generation proceed even if rerank confidence is not strong
        if answerability.get("can_answer", False):
            return {
                "confidence": "medium",
                "action": "generate_cautiously",
                "reason": "Answerability indicates sufficient evidence."
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

        combined_fit = 0.85 * best_score + 0.15 * best_overlap

        if combined_fit >= 0.605:
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
        skip_retrieval_cache: bool = True,
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

        # Modern systems usually classify the query before pasing it into the domain gate for retrieval
        intent_info = self._classify_query_intent(query)

        # This is the actual gate check before retrieving
        # If query looks OOD then no expansion, no retrieval, no reranking and no retry loop

        # Timing the domain gate separately to understand its latency impact on the overall system
        # and to ensure it remains lightweight as intended. We want to catch OOD queries early
        # without adding significant overhead to in-domain queries.
        domain_gate_start = time.perf_counter()

        domain_gate = self._domain_gate_preflight(query)


        domain_gate_latency_ms = round(
            (time.perf_counter() - domain_gate_start) * 1000,
            2
        )

        domain_gate["latency_ms"] = domain_gate_latency_ms

        if domain_gate and domain_gate["is_ood"]:
            empty_diag = {
                "query": query,
                "expanded_queries": [],
                "num_expanded_queries": 0,
                "query_intent": intent_info["intent"],
                "query_intent_reason": intent_info["reason"],
                "retrieval_strategy": "skipped",
                "retrieval_strategy_details": {
                    "strategy": "skipped",
                    "reason": "Domain gate detected an out-of-domain query.",
                    "signals": {},
                    "params": {},
                },
                "domain_gate": domain_gate,
                "duplicate_stats": {
                    "total_candidates": 0,
                    "unique_candidates": 0,
                    "duplicate_candidates": 0,
                    "duplicate_rate": 0.0,
                },
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
                "answerability": {
                    "score": 0.0,
                    "label": "low",
                    "can_answer": False,
                    "coverage_score": 0.0,
                    "semantic_score": 0.0,
                    "covered_terms": [],
                    "query_terms": [],
                    "reason": "No in-domain evidence was retrieved.",
                },
                "lexical_overlap": {
                    "average_lexical_overlap": 0.0,
                    "best_lexical_overlap": 0.0,
                    "per_chunk": [],
                },
                "metadata_coverage": {
                    "merged": {},
                    "reranked": {},
                    "context": {},
                },
                "source_diversity": {
                    "merged": {},
                    "reranked": {},
                    "context": {},
                },
                "per_query_results": [],
                "merged_results": {
                    "count": 0,
                    "sources": [],
                    "source_distribution": {},
                    "top_scores": [],
                    "top_multi_query_rrf_scores": [],
                },
                "merged_chunks": [],
                "reranked_results": {
                    "count": 0,
                    "sources": [],
                    "source_distribution": {},
                    "top_scores": [],
                    "top_rerank_scores": [],
                },
                "reranked_chunks": [],
                "context_summary": {
                    "count": 0,
                    "sources": [],
                    "source_distribution": {},
                },
                "latency": {
                    "domain_gate_latency_ms": domain_gate_latency_ms,
                    "retrieval_pipeline_latency_ms": 0.0,
                    "total_search_latency_ms": domain_gate_latency_ms,
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

            ood_result = {
                "used_retry": False,
                "confidence_route": {
                    "confidence": "low",
                    "action": "retry_or_abstain",
                    "reason": "Domain gate detected an out-of-domain query.",
                },
                "decision_trace": [],
                "chosen_final_k": 0,
                "chosen_context_top_k": 0,
                "retry_params": None,
                "initial_result": empty_result,
                "final_result": empty_result,
            }

            self._log_query_metrics(
                query,
                ood_result,
            )

            return ood_result

        # Time full initial retrieval pipeline
        initial_retrieval_start = time.perf_counter()

        initial_result = self.search_with_diagnostics(
            skip_retrieval_cache=skip_retrieval_cache,
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



        initial_retrieval_latency_ms = round(
            (time.perf_counter() - initial_retrieval_start) * 1000,
            2
        )

        # Attach gate info to the initial result so it always shows up in diagnostics
        initial_result["diagnostics"]["domain_gate"] = domain_gate

        # Store Initial Retrieval Latency
        initial_result["diagnostics"].setdefault("latency", {}).update({
            "domain_gate_latency_ms": domain_gate_latency_ms,
            "retrieval_pipeline_latency_ms": initial_retrieval_latency_ms,
        })

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
            healthy_result = {
                "chosen_final_k": chosen_final_k,
                "chosen_context_top_k": chosen_context_top_k,
                "used_retry": False,
                "confidence_route": initial_route,
                "retry_params": None,
                "initial_result": initial_result,
                "final_result": initial_result,
            }

            self._log_query_metrics(
                query,
                healthy_result,
            )

            return healthy_result

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

        # Measure retry cost
        retry_start = time.perf_counter()

        print("\nRETRY EXECUTION")
        print(f"retrieve_k={retry_policy['retrieve_k']}")
        print(f"final_k={chosen_final_k}")
        print(f"num_queries={retry_policy['num_queries']}")
        print("skip_retrieval_cache=True")

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
            skip_retrieval_cache=True
        )

        # We want proof that retry is actually doing something different than the initial retrieval and not just returning 
        # the same results due to cache or some other issue, so we compute the overlap between the initial retrieved 
        # chunks and the retry retrieved chunks as a sanity check. If there is very high overlap then it could 
        # indicate an issue with the retry logic.

        initial_chunks = initial_result["diagnostics"]["reranked_chunks"]
        retry_chunks = retry_result["diagnostics"]["reranked_chunks"]

        initial_texts = {
            c.get("text", "")[:100]
            for c in initial_chunks
        }

        retry_texts = {
            c.get("text", "")[:100]
            for c in retry_chunks
        }

        overlap = len(
            initial_texts & retry_texts
        )

        union = len(
            initial_texts | retry_texts
        )

        overlap_ratio = overlap / max(union, 1)

        print("\nRETRY OVERLAP")
        print(f"overlap_ratio={overlap_ratio:.2f}")

        initial_merged = {
            c.get("text", "")[:100]
            for c in initial_result["diagnostics"]["merged_chunks"]
        }

        retry_merged = {
            c.get("text", "")[:100]
            for c in retry_result["diagnostics"]["merged_chunks"]
        }

        overlap_on_merged = len(initial_merged & retry_merged)

        union_merged = len(initial_merged | retry_merged)

        overlap_merged_ratio = overlap_on_merged / max(union_merged, 1)

        print("\nRETRY OVERLAP ON MERGED CHUNKS")
        print(f"overlap_merged_ratio={overlap_merged_ratio:.2f}")

        print(
            f"\nINITIAL ANSWERABILITY SCORE: {initial_result['diagnostics']['answerability']['score']}"
        )

        print(
            f"RETRY ANSWERABILITY SCORE: {retry_result['diagnostics']['answerability']['score']}"
        )

        retry_latency_ms = round(
            (time.perf_counter() - retry_start) * 1000,
            2
        )

        retry_result["diagnostics"]["domain_gate"] = domain_gate

        # Store Retry Latency
        retry_result["diagnostics"].setdefault("latency", {}).update({
            "domain_gate_latency_ms": domain_gate_latency_ms,
            "retry_pipeline_latency_ms": retry_latency_ms,
            "retry_latency_ms": retry_latency_ms,
        })

        final_route = self._choose_confidence_route(retry_result["diagnostics"])

        retry_output = {
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
            "retry_latency_ms": retry_latency_ms,
        }

        self._log_query_metrics(
            query,
            retry_output,
        )

        return retry_output


    # Observability: log one event per query with key retrieval signals and the final confidence route decision for monitoring and analysis

    def _log_query_metrics(
        self,
        query,
        adaptive_result,
    ):
        """
        Store one observability event per query.
        """

        print("\nDEBUG: _log_query_metrics CALLED")


        initial_diag = (
            adaptive_result["initial_result"]
            .get("diagnostics", {})
        )

        final_diag = (
            adaptive_result["final_result"]
            .get("diagnostics", {})
        )

        initial_answerability = initial_diag.get(
            "answerability",
            {}
        )

        final_answerability = final_diag.get(
            "answerability",
            {}
        )

        initial_relevance = initial_diag.get(
            "relevance_confidence",
            {}
        )

        final_relevance = final_diag.get(
            "relevance_confidence",
            {}
        )

        initial_diversity = initial_diag.get(
            "diversity_confidence",
            {}
        )

        final_diversity = final_diag.get(
            "diversity_confidence",
            {}
        )

        initial_health = initial_diag.get(
            "retrieval_health",
            {}
        )

        final_health = final_diag.get(
            "retrieval_health",
            {}
        )

        domain_gate = final_diag.get(
            "domain_gate",
            {}
        )

        confidence_route = adaptive_result.get(
            "confidence_route",
            {}
        )


        initial_answerability_score = (
            initial_answerability.get(
                "score",
                0.0
            )
        )

        final_answerability_score = (
            final_answerability.get(
                "score",
                0.0
            )
        )

        answerability_delta = round(
            final_answerability_score
            - initial_answerability_score,
            3
        )

        initial_relevance_score = (
            initial_relevance.get(
                "score",
                0.0
            )
        )

        final_relevance_score = (
            final_relevance.get(
                "score",
                0.0
            )
        )

        relevance_delta = round(
            final_relevance_score
            - initial_relevance_score,
            3
        )

        initial_diversity_score = (
            initial_diversity.get(
                "score",
                0.0
            )
        )

        final_diversity_score = (
            final_diversity.get(
                "score",
                0.0
            )
        )

        diversity_delta = round(
            final_diversity_score
            - initial_diversity_score,
            3
        )

        latency_info = final_diag.get("latency", {})

        query_expansion_latency_ms = latency_info.get(
                "query_expansion_latency_ms",
                0.0
            )

        retrieval_latency_ms = latency_info.get(
            "retrieval_latency_ms",
            0.0
        )

        fusion_latency_ms = latency_info.get(
            "fusion_latency_ms",
            0.0
        )

        rerank_latency_ms = latency_info.get(
            "rerank_latency_ms",
            0.0
        )

        diagnostics_latency_ms = latency_info.get(
            "diagnostics_latency_ms",
            0.0
        )

        total_search_latency_ms = latency_info.get(
            "total_search_latency_ms",
            0.0
        )

        total_latency = max(
            total_search_latency_ms,
            1
        )

        retrieval_pct = round(
            retrieval_latency_ms / total_latency,
            3
        )

        fusion_pct = round(
            fusion_latency_ms / total_latency,
            3
        )

        rerank_pct = round(
            rerank_latency_ms / total_latency,
            3
        )

        diagnostics_pct = round(
            diagnostics_latency_ms / total_latency,
            3
        )

        rerank_candidates_count = latency_info.get(
            "rerank_candidates_count",
            0
        )

        latency_per_candidate_ms = latency_info.get(
            "latency_per_candidate_ms",
            0.0
        )



        event = {

            # Query
            "query": query,

            # Routing
            "intent":
                final_diag.get(
                    "query_intent",
                    "unknown"
                ),

            "strategy":
                final_diag.get(
                    "retrieval_strategy",
                    "unknown"
                ),

            # Domain Gate
            "is_ood":
                domain_gate.get(
                    "is_ood",
                    False
                ),

            "domain_fit":
                domain_gate.get(
                    "combined_fit",
                    0.0
                ),

            "domain_gate_confidence":
                domain_gate.get(
                    "confidence",
                    "unknown"
                ),

            "domain_gate_action":
                domain_gate.get(
                    "action",
                    "unknown"
                ),

            "expanded_queries_count":
                len(
                    final_diag.get(
                        "expanded_queries",
                        []
                    )
                ),

            "merged_candidates":
                len(
                    final_diag.get(
                        "merged_chunks",
                        []
                    )
                ),

            "reranked_candidates":
                len(
                    final_diag.get(
                        "reranked_chunks",
                        []
                    )
                ),

            "context_chunks":
                len(
                    adaptive_result["final_result"]
                    .get(
                        "context_chunks",
                        []
                    )
                ),

            # Controller
            "used_retry":
                adaptive_result.get(
                    "used_retry",
                    False
                ),

            # Cache

            "query_cache_hit":
                final_diag.get(
                    "cache",
                    {}
                ).get(
                    "query_cache_hit",
                    False
                ),

            "query_cache_miss":
                not final_diag.get(
                    "cache",
                    {}
                ).get(
                    "query_cache_hit",
                    False
                ),

            "query_cache_latency_saved_ms":
                final_diag.get(
                    "cache",
                    {}
                ).get(
                    "query_expansion_latency_ms",
                    0.0
                )
                if final_diag.get(
                    "cache",
                    {}
                ).get(
                    "query_cache_hit",
                    False
                )
                else 0.0,

            # TODO: done
            # Replace with actual saved latency from 2127
            # stored alongside cache entry.

            "retrieval_cache_hit":
                final_diag.get(
                    "cache",
                    {}
                ).get(
                    "retrieval_cache_hit",
                    False
                ),

            "retrieval_cache_miss":
                not final_diag.get(
                    "cache",
                    {}
                ).get(
                    "retrieval_cache_hit",
                    False
                ),

            "retrieval_cache_latency_saved_ms": 0.0,

            "confidence_action":
                confidence_route.get(
                    "action",
                    "unknown"
                ),


            "query_expansion_latency_ms":
                query_expansion_latency_ms,

            "retrieval_latency_ms":
                retrieval_latency_ms,

            "fusion_latency_ms":
                fusion_latency_ms,

            "rerank_latency_ms":
                rerank_latency_ms,

            "diagnostics_latency_ms":
                diagnostics_latency_ms,

            "total_search_latency_ms":
                total_search_latency_ms,

            "retrieval_pct":
                retrieval_pct,

            "fusion_pct":
                fusion_pct,

            "rerank_pct":
                rerank_pct,

            "diagnostics_pct":
                diagnostics_pct,

            "rerank_candidates_count":
                rerank_candidates_count,

            "latency_per_candidate_ms":
                latency_per_candidate_ms,

            "domain_gate_latency_ms":
                domain_gate.get("latency_ms", 0.0),

            "retry_latency_ms":
                latency_info.get(
                    "retry_latency_ms",
                    adaptive_result.get("retry_latency_ms", 0.0)
                ),

            "retry_reasons": [
                step.get("reason", "unknown")
                for step in adaptive_result.get(
                    "decision_trace",
                    []
                )
                if isinstance(step, dict)
            ],





            # Initial State
            "initial_health":
                initial_health.get(
                    "severity",
                    "unknown"
                ),

            "initial_answerability":
                initial_answerability_score,

            "initial_risk_score":
                initial_health.get(
                    "risk_score",
                    0.0
                ),

            # Final State
            "final_health":
                final_health.get(
                    "severity",
                    "unknown"
                ),

            "final_answerability":
                final_answerability_score,

            "final_risk_score":
                final_health.get(
                    "risk_score",
                    0.0
                ),

            # Improvement
            "answerability_delta":
                answerability_delta,

            "retry_improved_answerability":
                answerability_delta > 0,

            "retry_improved_relevance":
                relevance_delta > 0,

            "retry_improved_relevance":
                relevance_delta > 0,

            "initial_relevance":
                initial_relevance_score,

            "final_relevance":
                final_relevance_score,

            "relevance_delta":
                relevance_delta,

            "initial_diversity":
                initial_diversity_score,

            "final_diversity":
                final_diversity_score,

            "diversity_delta":
                diversity_delta,

            # Final Decision
            "final_confidence":
                confidence_route.get(
                    "confidence",
                    "unknown"
                ),

        
        }

        self.metrics_logger.log(event)


    def print_diagnostics(self, diag):
        print("\n==================================================")
        print("QUERY ANALYSIS")
        print("==================================================")


        print("\nORIGINAL QUERY:")
        print(diag["query"])

        print("\nQUERY INTENT:")
        print(f"  Intent: {diag.get('query_intent')}")
        print(f"  Reason: {diag.get('query_intent_reason')}")

        print("\nEXPANDED QUERIES:")
        for i, q in enumerate(diag["expanded_queries"], start=1):
            print(f"  [{i}] {q}")

        print("\nRETRIEVAL STRATEGY:")

        strategy = diag.get("retrieval_strategy", "unknown")
        strategy_details = diag.get("retrieval_strategy_details", {})

        print(f"  Strategy: {strategy}")
        print(f"  Reason: {strategy_details.get('reason', '')}")

        print("\n==================================================")
        print("RETRIEVAL PIPELINE")
        print("==================================================")

        print("\nDUPLICATE STATS:")
        dup = diag["duplicate_stats"]
        print(f"  Total candidates: {dup['total_candidates']}")
        print(f"  Unique candidates: {dup['unique_candidates']}")
        print(f"  Duplicate candidates: {dup['duplicate_candidates']}")
        print(f"  Duplicate rate: {dup['duplicate_rate']:.4f}")

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

        print("\n==================================================")
        print("MERGED RESULTS")
        print("==================================================")

        merged = diag["merged_results"]

        print(f"Merged Candidate Count: {merged['count']}")

        print("\nMerged Source Distribution:")
        for source, count in merged["source_distribution"].items():
            print(f"  {source}: {count}")

        # print("\nTop Multi-Query RRF Scores:")
        # print(merged["top_multi_query_rrf_scores"])

        # print("\nTOP 20 BEFORE RERANK")

        # fused_results = diag.get("merged_chunks", [])
        # reranked_chunks = diag.get("reranked_chunks", [])

        # print("\nPIPELINE COUNTS")
        # print(f"  merged_chunks   : {len(fused_results)}")
        # print(f"  reranked_chunks : {len(reranked_chunks)}")

        target_source = "REACT"
        merged_chunks = diag.get("merged_chunks", [])
        source_chunks = [
            c for c in merged_chunks
            if target_source.lower() in c.get("source", "unknown").lower()
        ]

        print(f"\nMerged Chunks for source containing '{target_source}' ({len(source_chunks)}):")
        for i, chunk in enumerate(source_chunks, start=1):
            print(f"\n[{i}] Source: {chunk.get('source', 'unknown')}")
            print(chunk.get("text", ""))


        print("\n==================================================")
        print("RERANKED RESULTS")
        print("==================================================")

        reranked = diag["reranked_results"]
        reranked_chunks = diag["reranked_chunks"]

        for i, chunk in enumerate(reranked_chunks[:6], start=1):

            source = chunk.get("source", "unknown")
            score = chunk.get("rerank_score", 0.0)

            text = chunk.get("text", "")
            preview = text[:300].replace("\n", " ")

            print(f"\n[{i}]")
            print(f"Source: {source}")
            print(f"Rerank Score: {score:.4f}")
            print(preview)

        print(f"Reranked Count: {reranked['count']}")

        print("\nReranked Source Distribution:")
        for source, count in reranked["source_distribution"].items():
            print(f"  {source}: {count}")

        print("\nTop Rerank Scores:")
        print(reranked["top_rerank_scores"])

        # print("\n==================================================")
        # print("RERANK ANALYSIS")
        # print("==================================================")

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

        print("\n==================================================")
        print("RETRIEVAL EVALUATION")
        print("==================================================")

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

        print("\nANSWERABILITY ESTIMATION:")

        answerability = diag.get("answerability", {})

        print(f"  Score: {answerability.get('score', 0.0)}")
        print(f"  Label: {answerability.get('label', 'unknown')}")
        print(f"  Can Answer: {answerability.get('can_answer', False)}")
        print(f"  Coverage Score: {answerability.get('coverage_score', 0.0)}")
        print(f"  Semantic Score: {answerability.get('semantic_score', 0.0)}")

        print("  Query Terms:")
        for term in answerability.get("query_terms", []):
            print(f"    - {term}")

        print("  Covered Terms:")
        for term in answerability.get("covered_terms", []):
            print(f"    - {term}")

        print(f"  Reason: {answerability.get('reason', '')}")

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

        # print("\n==================================================")
        # print("RETRIEVAL OBSERVABILITY")
        # print("==================================================")

        # print("\nMETADATA COVERAGE:")

        # for stage_name, stage_cov in diag["metadata_coverage"].items():
        #     print(f"\n{stage_name.upper()} CHUNKS:")

        #     for field, stats in stage_cov.items():
        #         print(
        #             f"  {field}: {stats['present']}/{stats['total']} "
        #             f"({stats['coverage_rate']:.4f})"
        #         )

        # print("\nSOURCE DIVERSITY:")

        # for stage_name, stage in diag["source_diversity"].items():
        #     print(f"\n{stage_name.upper()} CHUNKS:")

        #     print(f"  Total chunks: {stage['total_chunks']}")
        #     print(f"  Unique sources: {stage['unique_sources']}")
        #     print(f"  Dominant source: {stage['dominant_source']}")
        #     print(f"  Dominant source count: {stage['dominant_source_count']}")
        #     print(f"  Dominant ratio: {stage['dominant_ratio']:.4f}")
        #     print(f"  Top-3 source count: {stage['top_k_count']}")
        #     print(f"  Top-3 source ratio: {stage['top_k_ratio']:.4f}")
        #     print(f"  Balance: {stage['balance']}")

        #     print("  Source distribution:")

        #     for source, count in stage["source_distribution"].items():
        #         print(f"    {source}: {count}")



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

        relevance = adaptive_result["final_result"]["diagnostics"].get("relevance_confidence", {})
        diversity = adaptive_result["final_result"]["diagnostics"].get("diversity_confidence", {})

        answerability = adaptive_result["final_result"]["diagnostics"].get(
            "answerability",
            {}
        )

        strategy = adaptive_result["final_result"]["diagnostics"].get(
            "retrieval_strategy", "unknown"
        )

        strategy_details = adaptive_result["final_result"]["diagnostics"].get(
            "retrieval_strategy_details", {}
        )

        domain_gate = adaptive_result["final_result"]["diagnostics"].get("domain_gate", {})

        print("\n==================================================")
        print("ADAPTIVE CONTROLLER")
        print("==================================================")

        print("\nDOMAIN GATE / OOD DETECTION")

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
        else:
            print("  domain gate: not run")

        intent = adaptive_result["final_result"]["diagnostics"].get(
            "query_intent",
            "unknown"
        )

        intent_reason = adaptive_result["final_result"]["diagnostics"].get(
            "query_intent_reason",
            ""
        )

        print("\nRETRIEVAL STRATEGY")
        print(f"  strategy: {strategy}")
        print(f"  reason: {strategy_details.get('reason', '')}")

        print("\nQUERY INTENT")
        print(f"  intent: {intent}")
        print(f"  reason: {intent_reason}")

        print("\n==================================================")
        print("CONFIDENCE ANALYSIS")
        print("==================================================")

        print("\nRELEVANCE CONFIDENCE")
        print(f"  score: {relevance.get('score', 0.0)}")
        print(f"  label: {relevance.get('label', 'unknown')}")
        print(f"  reason: {relevance.get('reason', '')}")

        print("\nDIVERSITY CONFIDENCE")
        print(f"  score: {diversity.get('score', 0.0)}")
        print(f"  label: {diversity.get('label', 'unknown')}")
        print(f"  reason: {diversity.get('reason', '')}")

        print("\nANSWERABILITY ESTIMATION")
        print(f"  score: {answerability.get('score', 0.0)}")
        print(f"  label: {answerability.get('label', 'unknown')}")
        print(f"  can_answer: {answerability.get('can_answer', False)}")
        print(f"  coverage_score: {answerability.get('coverage_score', 0.0)}")
        print(f"  semantic_score: {answerability.get('semantic_score', 0.0)}")

        print("  covered_terms:")
        for term in answerability.get("covered_terms", []):
            print(f"    - {term}")

        confidence_route = adaptive_result.get("confidence_route", {})

        print("\nCONFIDENCE ROUTE")
        print(f"  confidence: {confidence_route.get('confidence', 'unknown')}")
        print(f"  action: {confidence_route.get('action', 'unknown')}")
        print(f"  reason: {confidence_route.get('reason', '')}")

        print("\n==================================================")
        print("ADAPTIVE DECISION")
        print("==================================================")

        print(f"Used retry: {adaptive_result['used_retry']}")

        if adaptive_result["retry_params"] is not None:

            print("\nRetry params:")

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

        print("\nFINAL HEALTH")
        print(f"  Severity: {final_health.get('severity', 'unknown')}")
        print(f"  Risk score: {final_health.get('risk_score', 0.0)}")
        print(f"  Is weak: {final_health.get('is_weak', False)}")



## Additional test case for search_with_context i.e. retrieving results along with a prompt-ready context string
if __name__ == "__main__":
    multi_query_retriever = MultiQueryHybridRetriever()

    # For adaptive search testing, use queries that are more likely to trigger reason-aware retrieval
    test_queries = [

            # FACT LOOKUP / DEFINITIONS

            # "What is ReAct and how does it work?",
            # "What problem does Toolformer solve?",
            # "What is Reflexion?",
            # "What is Self-challenging?",
            # "What is LoRA?",
            # "What is QLoRA?",
            # "What is FlashAttention?",
            # "What is dense and sparse retrieval?",
            #  "What is the ReAct framework?",


            # # WHO / WHEN / WHERE

            # "Who proposed the ReAct framework?",
            # "Who introduced Reflexion?",
            # "Who are the authors of Toolformer?",
            # "Who invented transformers?",
            # "When was LoRA introduced?",
            # "Where was ReAct published?",
            # "Where was QLoRA published?",

            # # EXPLANATION

            # "Explain Reflexion like I am a beginner.",
            # "What is agent memory and why is it important?",
            # "How does Tree of Thoughts work?",
            # "Explain Graph of Thoughts.",
            # "What is PagedAttention in vLLM?",

            # # COMPARISON

            # "Compare ReAct and Reflexion.",
            # "Compare LoRA and QLoRA.",
            # "Compare RAPTOR and traditional RAG.",
            # "Compare Self-RAG and Corrective RAG.",
            # "Compare Toolformer and ReAct.",
            # "Compare AutoGen and CAMEL.",
            # "Compare multi-agent systems and single-agent systems.",

            # # HOW TO

            # "How can Reflexion improve an AI system?",
            # "How do multi-agent systems coordinate tasks?",
            # "How does Toolformer enable tool usage?",
            # "How can LoRA be used to fine-tune an LLM?",
            # "How would I deploy an LLM using vLLM?",
            # "How would I build an adaptive RAG pipeline?",

            # # MULTI-HOP / SYNTHESIS

            # "How can ReAct and Toolformer be combined to build a tool-using agent?",
            # "What role does memory play in Reflexion-based agents?",
            # "How do retrieval quality and reranking affect agent performance?",
            # "How do RAPTOR and Self-RAG improve retrieval quality?",
            # "How would you design a production-grade agentic RAG system?",
            # "How would you architect an AI assistant using RAG, ReAct, and Reflexion?",

            # # EXPLORATION / REASONING

            # "What are the major limitations of ReAct?",
            # "What are the future directions of multi-agent AI systems?",
            # "What challenges arise when deploying agentic AI?",
            # "What techniques improve reliability in AI agents?",

            # # TIMELINE / FACT VERIFICATION

            # "Which came first, ReAct or Reflexion?",
            # "Which retrieval methods were proposed after the original RAG paper?",
            # "Did ReAct introduce tool usage in LLMs?",
            # "Is RAPTOR a retrieval method or a reranking method?",

            # # OOD

            # "What is the capital of India?"

]

    # test_queries = [
    #     "Ashish Vaswani",
    #     "authors of attention is all you need",
    #     "attention is all you need authors",
    #     "who authored attention is all you need",
    # ]


    for query in test_queries:
        print("\n==================================================")
        print("ORIGINAL QUERY:")
        print(query)


        # For demonstration, we will call the adaptive search which includes diagnostics and potential retry logic
        result = multi_query_retriever.adaptive_search_with_retry(
            query,
            num_queries=4,
            retrieve_k=40,
            final_k=10,
            context_top_k=5,
        )


         # --------------------------------------------------
        # ADAPTIVE RETRIEVAL + DIAGNOSTICS
        # --------------------------------------------------

        multi_query_retriever.print_adaptive_result(result)
        print("\nRETRIEVAL DIAGNOSTICS")
        multi_query_retriever.print_diagnostics(
            result["final_result"]["diagnostics"]
        )

        # --------------------------------------------------
        # CONTEXT CONSTRUCTION
        # --------------------------------------------------

        generator = Generator()

        original_context = result["final_result"]["context"]


        compressed_context = generator.compress_context(
            query=query,
            context=result["final_result"]["context"],
            confidence_route=result.get("confidence_route", {}),
        )



        print("\n==================================================")
        print("BEFORE COMPRESSION")
        print("\n==================================================")
        print(original_context)

        print("\n==================================================")
        print("AFTER COMPRESSION / PROMPT-READY CONTEXT")
        print("\n==================================================")
        print(compressed_context)


        print("\n=== LENGTH COMPARISON ===")

        original_len = len(original_context)
        compressed_len = len(compressed_context)

        reduction = original_len - compressed_len
        reduction_pct = (reduction / original_len * 100) if original_len > 0 else 0

        print(f"Original length   : {original_len}")
        print(f"Compressed length : {compressed_len}")
        print(f"Reduction         : {reduction}")
        print(f"Reduction %       : {reduction_pct:.2f}%")


        # Use the compressed context for answer generation to simulate the full pipeline
        # and see how the confidence route info can be passed to the generator for potential answer-level adjustments

        print("\n==================================================")
        print("GENERATION PREPARATION")
        print("==================================================")

        print("DYNAMIC CONTEXT SIZING POLICY")

        print(f"  chosen_final_k: {result.get('chosen_final_k')}")
        print(f"  chosen_context_top_k: {result.get('chosen_context_top_k')}")
        print(f"  retrieved_chunks: {len(result['final_result']['reranked_chunks'])}")
        print(f"  prompt_chunks: {len(result['final_result']['context_chunks'])}")

        # --------------------------------------------------
        # FINAL RERANKED RESULTS
        # --------------------------------------------------

        final_result = result["final_result"]

        print("\n==================================================")
        print("FINAL RERANKED RESULTS")
        print("==================================================")

        for i, c in enumerate(
            final_result["reranked_chunks"][:5],
            start=1,
        ):

            print(f"\n[{i}] source={c.get('source', 'unknown')}")

            if "rerank_score" in c:
                print(
                    f"rerank_score="
                    f"{c['rerank_score']:.4f}"
                )

            if "multi_query_rrf_score" in c:
                print(
                    f"multi_query_rrf_score="
                    f"{c['multi_query_rrf_score']:.4f}"
                )

            print(c.get("text", "")[:300])

            print("-" * 80)

        # --------------------------------------------------
        # GENERATION
        # --------------------------------------------------

        answer = generator.generate(
            query=query,
            context=compressed_context,
            confidence_route=result.get("confidence_route", {}),
            already_compressed=True,
    )

        print("\n==================================================")
        print("FINAL LLM ANSWER")
        print("==================================================")

        print(answer)

# if __name__ == "__main__":

#     mq = MultiQueryHybridRetriever()

#     hybrid_test_result = mq.retriever.hybrid_search(
#     "What is the ReAct framework?",
#     retrieve_k=40, final_k=10, k=60,
#     dense_weight=0.5, sparse_weight=0.5,
#     adaptive_weights=False,
#     min_dense_similarity=0.0, min_bm25_score=0.0,
#     max_per_source=None,
# )

#     for i, chunk in enumerate(hybrid_test_result, start=1):
#         print(f"[{i}] source={chunk.get('source')} chunk_id={chunk.get('chunk_id')} "
#             f"page={chunk.get('page')} text_start={chunk.get('text','')[:60]!r}")