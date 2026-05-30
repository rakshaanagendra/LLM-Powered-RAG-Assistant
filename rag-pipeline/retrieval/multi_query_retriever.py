import re
import json
import ollama


class MultiQueryRetriever:
    def __init__(self, model_name="qwen2.5:3b", client=None):
        self.model_name = model_name
        self.client = client

    def _clean_text(self, text):
        text = text.strip()
        text = text.strip('"').strip("'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _deduplicate_queries(self, queries):
        seen = set()
        unique = []

        for q in queries:
            cleaned = self._clean_text(q)

            if not cleaned:
                continue

            lowered = cleaned.lower()

            if lowered not in seen:
                unique.append(cleaned)
                seen.add(lowered)

        return unique

    def generate_queries(self, query, num_queries=4):
        query = self._clean_text(query)

        prompt = f"""
You are a retrieval query expansion system for RAG.

Your job is to produce a small set of highly targeted search queries that help document retrieval.

Rules:
- Return ONLY a valid JSON list of strings.
- Do not use markdown.
- Do not answer the question.
- Do not explain anything.
- Keep each query short, specific, and retrieval-oriented.
- Prefer exact terms, technical keywords, abbreviations, and concise paraphrases.
- Avoid verbose, generic, or overly broad queries.
- Avoid repeating the same idea in different words.
- Do not include filler words unless they are essential.
- Generate at most {num_queries - 1} new queries.
- The original query will be added automatically.

Good examples:
[
  "dense retrieval",
  "sparse retrieval bm25",
  "hybrid retrieval reranking"
]

Bad examples:
[
  "how to improve retrieval quality",
  "alternative phrasings for the concept",
  "more information about this topic"
]

User Query:
{query}
""".strip()

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate short, precise, retrieval-focused search queries. "
                        "You prefer technical terms and concise variants."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            if self.client is None:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": 0.1,
                        "top_p": 0.8,
                    },
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                )

            content = response["message"]["content"]
            content = self._clean_text(content)

            # Try to extract a JSON list even if the model adds extra text
            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                content = match.group(0)

            generated_queries = json.loads(content)

            if not isinstance(generated_queries, list):
                return [query]

            generated_queries = [
                self._clean_text(q)
                for q in generated_queries
                if isinstance(q, str) and self._clean_text(q)
            ]

            generated_queries.insert(0, query)
            generated_queries = self._deduplicate_queries(generated_queries)

            return generated_queries[:num_queries]

        except Exception:
            return [query]


if __name__ == "__main__":
    retriever = MultiQueryRetriever()

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

        expanded_queries = retriever.generate_queries(query)

        print("\nGENERATED QUERIES:\n")

        for i, q in enumerate(expanded_queries, start=1):
            print(f"[{i}] {q}")