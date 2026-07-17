import operator
from typing import Annotated
from typing_extensions import TypedDict


# -----------------------------------------------------------------------
# Custom reducer for node_latencies
# Why: LangGraph overwrites dict fields by default — each node's return
# would replace the previous node's latency entry instead of merging.
# This reducer merges dicts: {**old, **new} combines both into one.
# Same concept as operator.add for lists — just for dicts.
# -----------------------------------------------------------------------
def merge_dicts(old: dict, new: dict) -> dict:
    return {**old, **new}


class MultiAgentState(TypedDict):
    # Input
    query: str

    # Research Agent output
    research_context: str
    sources: list[str]
    confidence: str        # "high", "medium", "low"
    action: str            # "generate", "generate_cautiously", "retry_or_abstain"
    answerable: bool
    retrieval_strategy: str
    source_type: str       # "rag", "web_search", or "general_knowledge" —
                            # tells writer/critic whether real external
                            # content exists to check the answer against

    # Writer/Critic Agent output
    critique: str
    final_answer: str
    answer_grounded: bool
    retry_count:int
    abstained: bool        # True only if abstain_node overwrote final_answer

    # Observability — appends across agents, never overwrites
    agent_log: Annotated[list[str], operator.add]

    # Per-node latency — merges across agents, never overwrites
    # Each node writes {"its_node_name": latency_ms}
    # merge_dicts combines them: {"research_node": x, "writer_node": y, "critic_node": z}
    node_latencies: Annotated[dict, merge_dicts]