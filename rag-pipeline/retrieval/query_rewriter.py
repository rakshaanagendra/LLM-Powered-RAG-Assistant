import re
from pathlib import Path

import ollama

from caching.cache_manager import CacheManager


class QueryRewriter:
    def __init__(self, model_name="qwen2.5:7b", client=None):
        self.model_name = model_name
        self.client = client
        cache_file = Path(__file__).resolve().parents[1] / "caching" / "query_rewrite_cache.json"
        self.cache = CacheManager(cache_file)

    def _clean_text(self, text):
        text = text.strip()
        text = text.strip('"').strip("'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _fallback_rewrite(self, query):
        return self._clean_text(query)

    def rewrite(self, query):
        query = self._clean_text(query)

        cache_key = query.lower().strip()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = f"""
You are an expert retrieval query optimizer for RAG systems.

Rewrite the user's query to maximize retrieval quality from:
- technical documentation like text files and PDFs
- research papers
- AI documentation
- engineering notes

Instructions:
- Preserve original meaning
- Add important technical keywords if useful
- Expand abbreviations when useful
- Make the query retrieval-oriented
- Improve semantic and lexical matching
- Do NOT answer the question
- Return ONLY the rewritten query

User Query:
{query}
""".strip()

        try:
            if self.client is None:
                import ollama

                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You rewrite search queries for retrieval.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "temperature": 0.0,
                    },
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You rewrite search queries for retrieval.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "temperature": 0.0,
                    },
                )

            rewritten_query = response["message"]["content"]
            rewritten_query = self._clean_text(rewritten_query)

            if not rewritten_query:
                return self._fallback_rewrite(query)

            self.cache.set(cache_key, rewritten_query)
            return rewritten_query

        except Exception:
            return self._fallback_rewrite(query)
        

if __name__ == "__main__":
    rewriter = QueryRewriter()

    query = "How does retrieval augmented generation reduce hallucinations in LLM systems?"

    rewritten_query = rewriter.rewrite(query)

    print("\nOriginal Query:")
    print(query)

    print("\nRewritten Query:")
    print(rewritten_query)