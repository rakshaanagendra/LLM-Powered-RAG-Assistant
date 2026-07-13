
import json
import re
from typing import Any, Dict, List, Optional

import ollama


class LLMValidator:
    """
    LLM-based RAG validator.

    Input:
        - query
        - answer
        - chunks

    Output:
        - structured judgment dictionary
    """

    def __init__(self, model_name: str = "phi3"):
        self.model = model_name

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            parts.append(f"[ID: {i} | Source: {source}]\n{text}")
        return "\n\n".join(parts)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Tries to recover JSON from model output even if the model adds extra text.
        """
        if not text:
            return None

        cleaned = text.strip()

        # Remove fenced code blocks if present
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try direct JSON parse first
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Try to extract the first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = cleaned[start : end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _normalize_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the output has the expected keys.
        """
        defaults = {
            "verdict": "fail",
            "score": 0.0,
            "groundedness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "unsupported_claims": [],
            "missing_points": [],
            "action": "abstain",
            "reason": "Validator output could not be parsed cleanly.",
        }

        if not isinstance(data, dict):
            return defaults

        out = defaults.copy()
        out.update(data)

        # Make sure list fields really are lists
        if not isinstance(out.get("unsupported_claims"), list):
            out["unsupported_claims"] = [str(out["unsupported_claims"])]
        if not isinstance(out.get("missing_points"), list):
            out["missing_points"] = [str(out["missing_points"])]

        # Clamp numeric fields
        for key in ["score", "groundedness", "relevance", "completeness"]:
            try:
                out[key] = float(out[key])
            except Exception:
                out[key] = 0.0

        return out

    def validate(self, query: str, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main validation method.

        Returns a structured dict like:
        {
            "verdict": "pass" or "fail",
            "score": 0.0 to 1.0,
            "groundedness": 0.0 to 1.0,
            "relevance": 0.0 to 1.0,
            "completeness": 0.0 to 1.0,
            "unsupported_claims": [...],
            "missing_points": [...],
            "action": "accept" / "repair" / "retrieve" / "abstain",
            "reason": "..."
        }
        """
        if not answer or not answer.strip():
            return {
                "verdict": "fail",
                "score": 0.0,
                "groundedness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "unsupported_claims": [],
                "missing_points": ["The answer is empty."],
                "action": "abstain",
                "reason": "Empty answer.",
            }

        if not chunks:
            return {
                "verdict": "fail",
                "score": 0.0,
                "groundedness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "unsupported_claims": ["No context was provided."],
                "missing_points": [],
                "action": "retrieve",
                "reason": "No chunks available for validation.",
            }

        context = self._build_context(chunks)

        prompt = f"""
You are a strict LLM validator for a RAG system.

Your job:
- Judge whether the ANSWER is supported ONLY by the CONTEXT.
- Use ONLY the context. Do not use outside knowledge.
- Check if the answer is relevant to the QUESTION.
- Check if the answer is complete ONLY relative to the QUESTION.
- Do NOT require extra background, performance discussion, examples, or extended explanations unless the user explicitly asked for them.
- Identify unsupported claims only.
- Missing points should be only things explicitly asked by the user but absent from the answer.
- If the answer correctly addresses the question, do not penalize it for not being longer.

Scoring:
- groundedness: 0.0 to 1.0
- relevance: 0.0 to 1.0
- completeness: 0.0 to 1.0
- score: overall quality from 0.0 to 1.0

Decision rules:
- verdict = "pass" if the answer is grounded and answers the question.
- action = "accept" if the answer is good enough to return.
- action = "repair" only if an explicitly requested part is missing or wrong.
- action = "retrieve" if the context is insufficient.
- action = "abstain" if the answer cannot be trusted.

Return ONLY valid JSON. No markdown. No extra text.

Required JSON schema:
{{
  "verdict": "pass" or "fail",
  "score": 0.0,
  "groundedness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "unsupported_claims": ["..."],
  "missing_points": ["..."],
  "action": "accept" or "repair" or "retrieve" or "abstain",
  "reason": "short explanation"
}}

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 500},
        )

        raw = response["message"]["content"].strip()
        parsed = self._extract_json(raw)

        if parsed is None:
            return {
                "verdict": "fail",
                "score": 0.0,
                "groundedness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "unsupported_claims": ["Validator output could not be parsed as JSON."],
                "missing_points": [],
                "action": "repair",
                "reason": raw[:300],
            }

        return self._normalize_result(parsed)


if __name__ == "__main__":
    validator = LLMValidator(model_name="phi3")

    sample_chunks = [
        {
            "source": "doc1.txt",
            "text": "Hybrid search combines dense search and sparse search."
        },
        {
            "source": "doc2.txt",
            "text": "Dense search uses embeddings, while sparse search uses keyword matching such as BM25."
        },
    ]

    sample_query = "What is hybrid search in RAG?"
    sample_answer = (
        "Hybrid search combines dense search and sparse search. "
        "Dense search uses embeddings, while sparse search uses keyword matching such as BM25."
    )

    result = validator.validate(sample_query, sample_answer, sample_chunks)
    print(f"Verdict: {result.get('verdict')}")
    print(f"Score: {result.get('score')}")
    print(f"Groundedness: {result.get('groundedness')}")
    print(f"Relevance: {result.get('relevance')}")
    print(f"Completeness: {result.get('completeness')}")
    print(f"Unsupported claims: {result.get('unsupported_claims')}")
    print(f"Missing points: {result.get('missing_points')}")
    print(f"Action: {result.get('action')}")
    print(f"Reason: {result.get('reason')}")