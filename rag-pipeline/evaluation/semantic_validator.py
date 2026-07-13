import re
from sentence_transformers import SentenceTransformer, util


class SemanticValidator:
    def __init__(self, model_name="all-MiniLM-L12-v2"): #threshold=0.70):
        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Similarity threshold
        #self.threshold = threshold

    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, sentence_embedding, chunk_embeddings, top_k=2):
        # Compute cosine similarity
        similarities = util.cos_sim(sentence_embedding, chunk_embeddings)

        # Convert to list
        scores = similarities.squeeze().tolist()

        # Sort descending
        sorted_scores = sorted(scores, reverse=True)

        # Take top-k
        top_k_scores = sorted_scores[:top_k]

        # Combine evidence
        combined_score = sum(top_k_scores)

        return combined_score, top_k_scores

    def validate(self, answer, chunks, top_k=2, combined_threshold=1.1):
        """
        Returns:
            valid_sentences, invalid_sentences, sentence_data
        """

        sentences = self._split_sentences(answer)

        if not sentences:
            return [], [], []

        # 🔥 Precompute chunk embeddings ONCE
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)

        valid_sentences = []
        invalid_sentences = []
        sentence_data = []  # 🔥 NEW: store detailed info

        for sentence in sentences:

            # ❌ Skip source/meta lines
            if sentence.strip().startswith("[Source") or sentence.strip().startswith("Sources"):
                continue

            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

            combined_score, top_scores = self._compute_similarity(
                sentence_embedding, chunk_embeddings, top_k=top_k
            )

            is_valid = combined_score >= combined_threshold

            # 🔥 store everything (VERY IMPORTANT)
            sentence_data.append({
                "sentence": sentence,
                "score": combined_score,
                "top_k_scores": top_scores,
                "valid": is_valid
            })

            if is_valid:
                valid_sentences.append(sentence)
            else:
                invalid_sentences.append({
                    "sentence": sentence,
                    "combined_score": combined_score,
                    "top_k_scores": top_scores
                })

        return valid_sentences, invalid_sentences, sentence_data

    def validate_answer(self, answer, chunks, top_k=2, combined_threshold=1.1, strict_citations=False):
        """
        Returns:
            (is_valid: bool, reason: str, details: dict)
        """

        sentences = self._split_sentences(answer)

        if not sentences:
            return False, "Empty answer", {}

        # 🔥 Precompute chunk embeddings ONCE
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)

        details = {
            "total_sentences": len(sentences),
            "supported_sentences": 0,
            "unsupported_sentences": [],
            "sentence_scores": []
        }

        is_valid = True

        for i, sentence in enumerate(sentences):
            # ❌ Skip source/meta lines
            if sentence.strip().startswith("[Source"):
                continue

            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

            combined_score, top_scores = self._compute_similarity(
                sentence_embedding, chunk_embeddings, top_k=top_k
            )

            is_supported = combined_score >= combined_threshold

            details["sentence_scores"].append({
                "sentence": sentence,
                "combined_score": combined_score,
                "top_k_scores": top_scores,
                "supported": is_supported
            })

            if is_supported:
                details["supported_sentences"] += 1
            else:
                is_valid = False
                details["unsupported_sentences"].append({
                    "sentence": sentence,
                    "combined_score": combined_score,
                    "top_k_scores": top_scores
                })

        if is_valid:
            return True, "All sentences grounded", details
        else:
            return False, "Some sentences not grounded", details


# 🔥 TEST BLOCK
if __name__ == "__main__":
    validator = SemanticValidator()

    sample_chunks = [
        {
            "source": "langchain_retrievers.txt",
            "text": "A retriever in RAG systems fetches relevant documents based on a query using keyword or semantic similarity."
        },
        {
            "source": "advanced_rag.txt",
            "text": "Retrievers can be categorized into sparse retrievers such as BM25 and dense retrievers such as BERT-based models."
        },
    ]

    sample_answer = (
        "Retrievers in RAG fetch relevant documents for a query. "
        "There are two types of retrievers: sparse and dense. "
        "Sparse retrievers like BM25 match keywords, while dense retrievers use embeddings."
    )

    is_valid, reason, details = validator.validate_answer(sample_answer, sample_chunks)

    print("\nValid:", is_valid)
    print("Reason:", reason)
    print("Details:", details)