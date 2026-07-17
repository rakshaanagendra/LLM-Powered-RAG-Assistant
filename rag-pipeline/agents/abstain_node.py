import time
from agents.state import MultiAgentState

ABSTAIN_MESSAGE = "I don't have enough verified information to answer this confidently."

def abstain_node(state: MultiAgentState) -> dict:
    """
    Runs only when the writer/critic loop exhausted retries without
    producing a grounded answer. Overwrites final_answer with a hard
    abstention message — the ungrounded draft is discarded, never shown.
    """
    node_start = time.time()
    node_latency_ms = round((time.time() - node_start) * 1000, 2)

    log_entry = (
        f"[AbstainNode] Query: '{state['query']}' | "
        f"Discarded ungrounded draft after {state.get('retry_count', 0)} retries | "
        f"Latency: {node_latency_ms}ms"
    )

    return {
        "final_answer": ABSTAIN_MESSAGE,
        "abstained": True,
        "node_latencies": {"abstain_node": node_latency_ms},
        "agent_log": [log_entry],
    }