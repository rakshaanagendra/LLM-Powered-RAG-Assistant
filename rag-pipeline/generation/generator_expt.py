import ollama
import re
import sys
from pathlib import Path

QUERY = "What is the capital of France?"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from evaluation.semantic_validator import SemanticValidator
from evaluation.llm_validator import LLMValidator
from retrieval.multi_query_hybrid import MultiQueryHybridRetriever

semantic_validator = SemanticValidator()

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "some", "all", "name", "list", "tell",
}

def _extract_terms(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in STOPWORDS}

class Generator:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model = model_name
        self.llm_validator = LLMValidator(model_name=model_name)

    # --------------------------
    # Sentence splitting
    # --------------------------
    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    #--------------------------
    # Query-aware filtering
    #--------------------------
    def _is_relevant(self, sentence, query_terms, min_overlap=1):
        sentence_terms = _extract_terms(sentence)
        overlap = query_terms.intersection(sentence_terms)
        return len(overlap) >= min_overlap

    #--------------------------
    # Concept extraction
    #--------------------------
    # def _extract_concepts(self, sentences):
    #     concepts = []

    #     for s in sentences:
    #         s_lower = s.lower()

    #         # detect patterns like:
    #         # "types: sparse and dense"
    #         # "types are sparse, dense"
    #         if "type" in s_lower:
    #             words = re.findall(r"[a-zA-Z]+", s_lower)

    #             for w in words:
    #                 if w not in ["types", "retrievers", "are"]:
    #                     concepts.append(w)

    #     return list(set(concepts))
        
    #--------------------------

    # Merging broken fragments 

    
    def _merge_fragments(self, sentences):
        merged = []
        buffer = ""

        for s in sentences:
            if len(s) < 5 or s.strip() in ["1.", "2.", "-", ":"]:
                buffer += " " + s
            else:
                if buffer:
                    merged.append(buffer.strip() + " " + s)
                    buffer = ""
                else:
                    merged.append(s)

        if buffer:
            merged.append(buffer.strip())

        return merged
    # --------------------------
    # GROUPING LOGIC (core idea)
    # --------------------------
    def _group_and_filter(self, answer, chunks, query, top_k=2, threshold=1.05):
        sentences = self._split_sentences(answer)
        sentences = self._merge_fragments(sentences)

        query_terms = _extract_terms(query)
        soft_threshold = 0.85

        if not sentences:
            return []

        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = semantic_validator.model.encode(chunk_texts, convert_to_tensor=True)

        kept = []
        i = 0

        while i < len(sentences):
            sentence = sentences[i]

            # 🔥 Query-aware filtering
            if not self._is_relevant(sentence, query_terms):
                i += 1
                continue

            if sentence.startswith("Sources"):
                i += 1
                continue

            sent_emb = semantic_validator.model.encode(sentence, convert_to_tensor=True)
            score, _ = semantic_validator._compute_similarity(sent_emb, chunk_embeddings, top_k)

            #  Case 1: strong sentence
            if score >= threshold and self._is_relevant(sentence, query_terms):
                kept.append(sentence)

            #  Case 2: borderline but still useful
            elif score >= 0.95 and self._is_relevant(sentence, query_terms):
                kept.append(sentence)

                #  NEW: absorb following weaker sentences (coverage fix)
                j = i + 1
                while j < len(sentences):
                    next_sentence = sentences[j]

                    next_emb = semantic_validator.model.encode(next_sentence, convert_to_tensor=True)
                    next_score, _ = semantic_validator._compute_similarity(
                        next_emb, chunk_embeddings, top_k
                    )

                    # allow weaker but still relevant sentences
                    if next_score >= soft_threshold and self._is_relevant(next_sentence, query_terms):
                        kept.append(next_sentence)
                        j += 1
                    else:
                        break

                i = j
                continue

            # Case 3: try grouping multiple sentences (FIXED)
            combined_text = sentence
            j = i + 1

            while j < len(sentences):
                candidate_text = combined_text + " " + sentences[j]

                comb_emb = semantic_validator.model.encode(candidate_text, convert_to_tensor=True)
                combined_score, _ = semantic_validator._compute_similarity(
                    comb_emb, chunk_embeddings, top_k
                )

                # ✅ NEW: allow grouping until threshold reached
                if combined_score >= threshold and self._is_relevant(combined_text, query_terms):
                    kept.extend(sentences[i:j+1])
                    i = j + 1
                    break

                combined_text = candidate_text
                j += 1
            else:
                i += 1

        return kept

    def _assign_citations(self, sentences, chunks):
        from sentence_transformers import util

        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = semantic_validator.model.encode(chunk_texts, convert_to_tensor=True)

        cited = []

        for sentence in sentences:
            sent_emb = semantic_validator.model.encode(sentence, convert_to_tensor=True)
            similarities = util.cos_sim(sent_emb, chunk_embeddings).squeeze().tolist()

            best_idx = similarities.index(max(similarities))
            source = chunks[best_idx]["source"]

            cited.append(f"{sentence} [Source: {source}]")

        return cited
    
    def _compute_confidence(self, sentence_data):
        if not sentence_data:
            return 0.0, "Low"

        total = len(sentence_data)
        supported = sum(1 for s in sentence_data if s["valid"])

        coverage = supported / total if total > 0 else 0

        scores = [s["score"] for s in sentence_data if s["valid"]]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if coverage >= 0.8 and avg_score >= 1.2:
            return 0.85, "High"
        elif coverage >= 0.5 and avg_score >= 1.0:
            return 0.60, "Medium"
        else:
            return 0.30, "Low"
            
    def _compute_metrics(self, sentence_data):
        total = len(sentence_data)
        supported = sum(1 for s in sentence_data if s["valid"])

        accuracy = supported / total if total > 0 else 0

        return {
            "total_sentences": total,
            "supported_sentences": supported,
            "grounding_accuracy": round(accuracy, 2)
        }

    def _print_validation_result(self, label, result):
        print(label)
        print(f"  Verdict: {result.get('verdict')}")
        print(f"  Score: {result.get('score')}")
        print(f"  Groundedness: {result.get('groundedness')}")
        print(f"  Relevance: {result.get('relevance')}")
        print(f"  Completeness: {result.get('completeness')}")
        print(f"  Unsupported claims: {result.get('unsupported_claims')}")
        print(f"  Missing points: {result.get('missing_points')}")
        print(f"  Action: {result.get('action')}")
        print(f"  Reason: {result.get('reason')}")

    def _print_confidence(self, label, confidence):
        print(f"{label} {confidence}")

    def _format_final_answer(self, cited_sentences):
        return "\n".join(cited_sentences)
    # --------------------------
    # GENERATION
    # --------------------------
    def generate(self, query, chunks):
        if not chunks:
            return "I don't know."

        context = "\n\n".join([
            f"[ID: {i} | Source: {c['source']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        ])

        prompt = f"""
You are a STRICT, FACTUAL, GROUNDED AI assistant.

Rules:
- Use ONLY the provided context.
- Do NOT introduce new facts.
- If the context does not support the answer, output exactly: I don't know.
- Answer ONLY what is asked.
- If the question asks for types, list only the types that are directly relevant to the question.
- Do NOT add advantages, background theory, or extra categories unless asked.
- Keep the answer concise but informative: 3 to 8 sentences max.
- Do NOT add bullet lists, headings, or commentary.
- Do NOT mention confidence, validation, or citations in the answer text.
- Prefer direct, plain English.

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 300}
        )

        answer = response["message"]["content"]

        # --------------------------
        # STEP 1: Validation
        # --------------------------
        valid_sentences, invalid_sentences, sentence_data = semantic_validator.validate(
            answer,
            chunks,
            top_k=2,
            combined_threshold=0.95
        )

        # --------------------------
        # STEP 2: Filtering + grouping
        # --------------------------
        filtered_sentences = self._group_and_filter(
            answer,
            chunks,
            query,
            top_k=2,
            threshold=0.95
        )

        # filtered_sentences = self._ensure_concept_coverage(
        #     filtered_sentences,
        #     self._split_sentences(answer)
        # )
        if filtered_sentences:
            # Remove duplicates while preserving order
            seen = set()
            deduped = []
            for s in filtered_sentences:
                if s not in seen:
                    deduped.append(s)
                    seen.add(s)
            filtered_sentences = deduped

            # Build plain-text candidate answer for LLM validation
            candidate_answer = " ".join(filtered_sentences)

            # STEP 3: LLM validation
            llm_result = self.llm_validator.validate(query, candidate_answer, chunks)
            self._print_validation_result("🧠 LLM Validator:", llm_result)

            # STEP 4: Decide what to do
            if llm_result["action"] in ["accept", "repair"]:
                cited = self._assign_citations(filtered_sentences, chunks)

                confidence_score, confidence_label = self._compute_confidence(sentence_data)
                metrics = self._compute_metrics(sentence_data)

                print("📊 Metrics:", metrics)
                print(f"📊 Confidence: {confidence_label} ({confidence_score:.2f})")

                final_answer = self._format_final_answer(cited)
                final_answer += f"\n\nConfidence: {confidence_label} ({confidence_score:.2f})"

                print("✅ Returning validated grounded answer.")
                return final_answer

#             elif llm_result["action"] == "repair":
#                 try:
#                     repair_prompt = f"""
# Rewrite the answer using ONLY the context.

# - Keep all supported information
# - Remove unsupported parts
# - Do NOT add new facts
# - Fix any missing important points if they are in the context

# QUESTION:
# {query}

# CONTEXT:
# {context}

# DRAFT ANSWER:
# {candidate_answer}

# VALIDATOR FEEDBACK:
# {llm_result}

# REWRITTEN ANSWER:
# """
#                     repair_response = ollama.chat(
#                         model=self.model,
#                         messages=[{"role": "user", "content": repair_prompt}],
#                         options={"temperature": 0.1, "num_predict": 300}
#                     )

#                     repaired_answer = repair_response["message"]["content"].strip()

#                     # Validate repaired answer again
#                     repaired_valid_sentences, repaired_invalid_sentences, repaired_sentence_data = semantic_validator.validate(
#                         repaired_answer,
#                         chunks,
#                         top_k=2,
#                         combined_threshold=1.1
#                     )

#                     repaired_check = self.llm_validator.validate(query, repaired_answer, chunks)
#                     self._print_validation_result("🧠 LLM Validator (repaired):", repaired_check)

#                     if repaired_check["action"] == "accept":
#                         repaired_sentences = self._split_sentences(repaired_answer)
#                         repaired_sentences = self._merge_fragments(repaired_sentences)
#                         repaired_sentences = self._ensure_concept_coverage(
#                             repaired_sentences,
#                             self._split_sentences(repaired_answer)
#                         )

#                         cited = self._assign_citations(repaired_sentences, chunks)
#                         confidence = self._compute_confidence(repaired_sentence_data)

#                         self._print_confidence("📊 Confidence (repaired):", confidence)
#                         print("✅ Returning repaired + validated answer.")
#                         return self._format_final_answer(cited)

#                 except Exception as e:
#                     print("Repair failed:", str(e))

            # If validator says retrieve/abstain or repair did not work
            print("⚠️ LLM validator rejected the answer.")
            return "I don't know."

        # --------------------------
        # STEP 5: Repair loop (fallback)
        # --------------------------
#         try:
#             repair_prompt = f"""
# Rewrite the answer using ONLY the context.

# - Keep all supported information
# - Remove unsupported parts
# - Do NOT add new facts

# CONTEXT:
# {context}

# ORIGINAL ANSWER:
# {answer}

# REWRITTEN ANSWER:
# """

#             repair_response = ollama.chat(
#                 model=self.model,
#                 messages=[{"role": "user", "content": repair_prompt}],
#                 options={"temperature": 0.1, "num_predict": 300}
#             )

#             repaired_answer = repair_response["message"]["content"]

#             # Validate repaired answer
#             valid_sentences, invalid_sentences, sentence_data = semantic_validator.validate(
#                 repaired_answer,
#                 chunks,
#                 top_k=2,
#                 combined_threshold=1.1
#             )

#             filtered_sentences = self._group_and_filter(
#                 repaired_answer,
#                 chunks,
#                 query,
#                 top_k=2,
#                 threshold=1.1
#             )

#             if filtered_sentences:
#                 cited = self._assign_citations(filtered_sentences, chunks)
#                 confidence = self._compute_confidence(sentence_data)

#                 print("📊 Confidence (repaired):", confidence)

#                 print("✅ Returning repaired + filtered answer.")
#                 return self._format_final_answer(cited)

#         except Exception as e:
#             print("Repair failed:", str(e))

        # --------------------------
        # FINAL SAFE FALLBACK
        # --------------------------
        print("⚠️ Returning 'I don't know.'")
        return "I don't know."

    #--------------------------
    #Ensure concept coverage (for future use in grouping)
    #--------------------------

    # def _ensure_concept_coverage(self, filtered_sentences, original_sentences):
    #     concepts = self._extract_concepts(filtered_sentences)

    #     text = " ".join(filtered_sentences).lower()
        
    #     # Track already-added sentences to avoid duplicates
    #     added_set = set(filtered_sentences)

    #     for concept in concepts:
    #         if concept not in text:
    #             for s in original_sentences:
    #                 if concept in s.lower() and s not in added_set:
    #                     filtered_sentences.append(s)
    #                     added_set.add(s)
    #                     break

    #     return filtered_sentences

if __name__ == "__main__":

    ## OLD FLOW WITHOUT MULTI-QUERY
    # retriever = Retriever()
    # reranker = Reranker()
    # generator = Generator()

    # retrieved = retriever.hybrid_search(QUERY, retrieve_k=30, final_k=30)
    # reranked = reranker.rerank(QUERY, retrieved, top_k=5)

    # answer = generator.generate(QUERY, reranked)

    # print("\n=== USER QUERY ===")
    # print(QUERY)

    # print("\n========== GENERATED ANSWER ==========\n")
    # print(answer)

    ## NEW FLOW WITH MULTI-QUERY
    multi_query_retriever = MultiQueryHybridRetriever()
    generator = Generator()

    result = multi_query_retriever.search_with_context(
        QUERY,
        num_queries=4,
        retrieve_k=30,
        final_k=10,
        context_top_k=5,
    )

    reranked = result["reranked_chunks"]

    print("\n=== USER QUERY ===")
    print(QUERY)

    print("\n === MULTI-QUERY HYBRID RETRIEVAL RESULT ===")
    print(f"Expanded Queries: {result['expanded_queries']}")
    print(f"Context Chunks: {result['context_chunks']}")


    print("\n=== RERANKED CHUNKS ===")
    for i, c in enumerate(reranked, start=1):
        print(f"[{i}] source={c.get('source', 'unknown')}")
        if "rerank_score" in c:
            print(f"rerank_score={c['rerank_score']:.4f}")
        if "multi_query_rrf_score" in c:
            print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")
        print(c.get("text", "")[:300])
        print("-" * 80)

    print("\n=== PROMPT-READY CONTEXT ===")
    print(result["context"])

    answer = generator.generate(QUERY, reranked[:5])

    print("\n========== GENERATED ANSWER ==========\n")
    print(answer)

## NEW pipeline

# Query
# ↓
# Multi-Query Expansion
# ↓
# Query Variants Generated
#    ├─ Original query
#    ├─ Semantic reformulations
#    ├─ Retrieval-oriented rewrites
#    └─ Vocabulary-expansion variants
# ↓
# Hybrid Retrieval for EACH Query
#    ├─ Dense Retrieval (Embeddings / Semantic Search)
#    └─ Sparse Retrieval (BM25 / Keyword Search)
# ↓
# Hybrid Score Fusion
#    ├─ Dense scores
#    ├─ Sparse scores
#    └─ Weighted hybrid ranking
# ↓
# Multi-Query Aggregation
# ↓
# Reciprocal Rank Fusion (RRF)
# ↓
# Deduplication + Source Balancing
# ↓
# Cross-Encoder Reranking
# ↓
# Top-K Reranked Chunks
# ↓
# Prompt-Ready Context Construction
# ↓
# LLM Generation (strict grounded prompt)
# ↓
# Sentence Split
# ↓
# Semantic Validation (embedding grounding check)
# ↓
# Query-Aware Filtering
# ↓
# Sentence Grouping + Dependency Preservation
# ↓
# Deduplication
# ↓
# Build Plain-Text Candidate Answer
# ↓
# LLM Validator
#    ├─ Groundedness
#    ├─ Relevance
#    ├─ Completeness
#    └─ Unsupported Claims
# ↓
# Decision:
#    ├─ ACCEPT
#    │    ↓
#    │ Assign citations
#    │ ↓
#    │ Compute confidence + metrics
#    │ ↓
#    │ RETURN grounded answer
#    │
#    ├─ REPAIR (currently optional / partially disabled) BEST NOT TO USE THIS BRANCH UNLESS FOR EDUCATIONAL PURPOSES, AS IT CAN LEAD TO UNPREDICTABLE RESULTS
#    │    ↓
#    │ Rewrite using validator feedback
#    │ ↓
#    │ Revalidate semantically
#    │ ↓
#    │ Revalidate with LLM judge
#    │ ↓
#    │ If valid → citations + confidence → RETURN
#    │
#    └─ FAIL / ABSTAIN
#         ↓
#         RETURN "I don't know."
# ↓
# Final Safe Fallback:
#    If no sufficiently grounded content survives validation
#    → RETURN "I don't know."





## OLD FLOW

# Query
# ↓
# Hybrid Retrieval (Dense + Sparse)
# ↓
# Reranking (Top relevant chunks)
# ↓
# LLM Generation (strict prompt)
# ↓
# Sentence Split
# ↓
# Semantic Validation (embedding-based grounding check)
# ↓
# Filtering + Grouping (remove weak/unsupported sentences, preserve dependencies)
# ↓
# Concept Coverage Enrichment
# ↓
# Build Plain-Text Candidate Answer
# ↓
# LLM Validator (checks groundedness, relevance, completeness, unsupported claims)
# ↓
# Decision:
#    ├─ PASS
#    │    ↓
#    │ Deduplication
#    │ ↓
#    │ Assign citations
#    │ ↓
#    │ Compute confidence + metrics
#    │ ↓
#    │ RETURN answer
#    │
#    ├─ REPAIR
#    │    ↓
#    │ Rewrite using context + validator feedback
#    │ ↓
#    │ Semantic Validation
#    │ ↓
#    │ LLM Validation again
#    │ ↓
#    │ If valid → citations + confidence → RETURN
#    │
#    └─ FAIL / ABSTAIN / RETRIEVE
#         ↓
#         RETURN "I don't know."
# ↓
# If no valid filtered content exists:
#    → Repair original answer
#    → Semantic Validation
#    → Filtering + Grouping
#    → If still invalid → "I don't know."