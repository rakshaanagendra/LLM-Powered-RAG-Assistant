import ollama


class Generator:
    """
    Simple confidence-aware generator.

    Responsibilities:
    - Build prompts
    - Call LLM
    - Apply confidence-aware generation behavior

    NOT responsible for:
    - Retrieval
    - Reranking
    - Validation
    - Retry logic
    """

    def __init__(self, model_name="qwen2.5:3b"):
        self.model = model_name

    def _build_prompt(self, query, context, confidence_level):
        """
        Build a prompt based on retrieval confidence.
        """

        if confidence_level == "high":

            instructions = """
                You are a factual RAG assistant.

                Rules:
                - Use ONLY the provided context.
                - Answer the user's question directly.
                - Do not invent information.
                - If the answer is not supported by the context, say:
                "I don't know."
                - Keep the answer concise and informative.
                """

        elif confidence_level == "medium":

            instructions = """
                You are a cautious factual RAG assistant.

                Rules:
                - Use ONLY the provided context.
                - Answer ONLY information explicitly stated in the context.
                - Do not infer, speculate, or fill gaps.
                - If any part of the answer is unsupported, say:
                "I don't know."
                - Be conservative.
                """

        else:
            instructions = """
                You are a factual assistant.

                Rules:
                - If evidence is insufficient, say:
                "I don't know."
                """

        prompt = f"""
{instructions}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        return prompt

    def _generate(self, prompt):
        """
        Single LLM call.
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0,
                "num_predict": 300
            }
        )

        return response["message"]["content"].strip()

    def generate(self, query, context, confidence_route):
        """
        Main generation entrypoint.
        """

        confidence = confidence_route.get("confidence", "low")
        action = confidence_route.get("action", "retry_or_abstain")

        print("\n==================================================")
        print("GENERATION ROUTING")
        print("==================================================")
        print(f"Confidence: {confidence}")
        print(f"Action: {action}")

        # LOW confidence -> abstain
        if action == "retry_or_abstain":
            return (
                "I do not have enough evidence in the knowledge base "
                "to answer this question."
            )

        prompt = self._build_prompt(
            query=query,
            context=context,
            confidence_level=confidence
        )

        answer = self._generate(prompt)

        return answer