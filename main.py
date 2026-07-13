import json
from fastapi.responses import StreamingResponse
from typing import Generator

from dotenv import load_dotenv
load_dotenv()

import sys
import time
import mlflow
from pathlib import Path
from typing import cast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------
# Path setup
# Why: main.py lives at project root, but multi_agent_graph.py is inside
# rag-pipeline/agents/. We add rag-pipeline to sys.path so Python can
# find it without relative import errors.
# -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PROJECT_ROOT / "rag-pipeline"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.multi_agent_graph import graph
from agents.state import MultiAgentState

mlflow.set_tracking_uri("sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix())
mlflow.set_experiment("production-rag-multi-agent")
# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------
app = FastAPI(
    title="Production RAG Multi-Agent System",
    description="A production-grade RAG and multi-agent AI system with retrieval, writing, and grounding evaluation.",
    version="1.0.0"
)

# -----------------------------------------------------------------------
# In-memory metrics store
# Why: simple counter to track usage stats exposed via /metrics
# This gets reset when the server restarts — MLflow will persist properly later
# -----------------------------------------------------------------------
metrics = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "grounded_count": 0,
    "ungrounded_count": 0,
}

# -----------------------------------------------------------------------
# Pydantic models
# Why: FastAPI uses these as contracts — it validates incoming JSON against
# QueryRequest automatically and rejects malformed requests before they
# reach the graph. QueryResponse defines exactly what the client gets back.
# -----------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str   # the user's query — only required field

class QueryResponse(BaseModel):
    answer: str
    context: str = ""
    confidence: str
    grounded: bool
    critique: str
    sources: list[str]
    latency_ms: float
    node_latencies: dict
    agent_log: list[str]

# -----------------------------------------------------------------------
# Streaming generator
# Why: graph.stream() yields state updates after each node completes.
# We format each update as an SSE chunk and yield it to the client.
# SSE format requires: data: {json}\n\n — the double newline signals
# end of one chunk to the client.
# -----------------------------------------------------------------------
def stream_pipeline(question: str) -> Generator[str, None, None]:

    start = time.time()
    final_state = {}

    yield f"data: {json.dumps({'type': 'start', 'message': 'Pipeline started'})}\n\n"

    try:
        for chunk in graph.stream(
            cast(MultiAgentState, {"query": question}),
            config={"configurable": {"stream_mode": True}},
            stream_mode=["updates", "messages"]
        ):
            # Both chunk types are now tuples: (event_type, payload)
            if not isinstance(chunk, tuple):
                continue

            event_type, payload = chunk

            # ---------------------------------------------------
            # Node update chunks
            # event_type == "updates"
            # payload == {"node_name": state_changes_dict}
            # ---------------------------------------------------
            if event_type == "updates" and isinstance(payload, dict):
                for node_name, state_update in payload.items():
                    # Merge all fields normally
                    for key, value in state_update.items():
                        if key == "node_latencies":
                            # Merge dicts instead of overwriting
                            if "node_latencies" not in final_state:
                                final_state["node_latencies"] = {}
                            final_state["node_latencies"].update(value)
                        elif key == "agent_log":
                            # Append lists instead of overwriting
                            if "agent_log" not in final_state:
                                final_state["agent_log"] = []
                            final_state["agent_log"].extend(value)
                        else:
                            final_state[key] = value

                    if node_name == "research_node":
                        yield f"data: {json.dumps({'type': 'node_complete', 'node': 'research_node', 'confidence': state_update.get('confidence', 'unknown'), 'retrieval_strategy': state_update.get('retrieval_strategy', 'unknown'), 'sources_count': len(state_update.get('sources', []))})}\n\n"

                    elif node_name == "writer_node":
                        yield f"data: {json.dumps({'type': 'node_complete', 'node': 'writer_node', 'answer_length': len(state_update.get('final_answer', ''))})}\n\n"

                    elif node_name == "critic_node":
                        yield f"data: {json.dumps({'type': 'node_complete', 'node': 'critic_node', 'grounded': state_update.get('answer_grounded', False), 'critique': state_update.get('critique', '')})}\n\n"

            # ---------------------------------------------------
            # Token chunks
            # event_type == "messages"
            # payload == (AIMessageChunk, metadata_dict)
            # Only stream writer_node tokens — skip internal LLM
            # calls from research and critic nodes
            # ---------------------------------------------------
            elif event_type == "messages" and isinstance(payload, tuple):
                message_chunk, metadata = payload
                node_name = metadata.get("langgraph_node", "")
                token = getattr(message_chunk, "content", "")

                if node_name == "writer_node" and token:
                    # Strip JSON scaffolding tokens — only yield
                    # tokens that are actual answer content
                    skip_tokens = ['{\n', '}', '{"', '"final_answer"',
                                   '"final', '_answer"', '": "', '":\n', '"\n}',
                                   ' "final_answer":', '"final_answer": "']
                    if token.strip() not in skip_tokens and token.strip() not in ['{', '}', '']:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    total_latency_ms = round((time.time() - start) * 1000, 2)

    yield f"data: {json.dumps({'type': 'done', 'answer': final_state.get('final_answer', ''), 'grounded': final_state.get('answer_grounded', False), 'sources': final_state.get('sources', []), 'latency_ms': total_latency_ms, 'node_latencies': final_state.get('node_latencies', {})})}\n\n"

# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Why: load balancers and cloud platforms (Render, Railway) ping this
    to know if the service is alive. Must return 200 OK quickly.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main inference endpoint.
    Accepts a question, runs the full multi-agent pipeline,
    returns the answer with metadata.
    """
    # Validate input is not empty
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Time the full pipeline
    start = time.time()

    try:
        result = graph.invoke(
            cast(MultiAgentState, {"query": request.question})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    total_latency_ms = round((time.time() - start) * 1000, 2)
    node_latencies = result.get("node_latencies", {})

    # Update in-memory metrics
    metrics["total_queries"] += 1
    metrics["total_latency_ms"] += total_latency_ms
    if result.get("answer_grounded"):
        metrics["grounded_count"] += 1
    else:
        metrics["ungrounded_count"] += 1

    # ------------------------------------------------------------------
    # MLflow GenAI trace
    # ------------------------------------------------------------------
    try:
        print("DEBUG: starting MLflow logging", flush=True)
        with mlflow.start_run():
            print("DEBUG: inside mlflow run", flush=True)
            mlflow.log_param("question", request.question[:250])
            mlflow.log_param("retrieval_strategy", result.get("retrieval_strategy", "unknown"))
            mlflow.log_param("action", result.get("action", "unknown"))
            mlflow.log_param("confidence", result.get("confidence", "unknown"))
            mlflow.log_metric("total_latency_ms", total_latency_ms)
            mlflow.log_metric("answer_grounded", int(result.get("answer_grounded", False)))
            mlflow.log_metric("sources_count", len(result.get("sources", [])))
            mlflow.log_metric("answer_length_chars", len(result.get("final_answer", "")))
            for node_name, latency in node_latencies.items():
                mlflow.log_metric(f"latency_{node_name}_ms", latency)
            print("DEBUG: MLflow logging complete", flush=True)
    except Exception as e:
        print(f"DEBUG: MLflow logging failed: {e}", flush=True)


    return QueryResponse(
        answer=result.get("final_answer", ""),
        # context=clean_context,
        confidence=result.get("confidence", "unknown"),
        grounded=result.get("answer_grounded", False),
        critique=result.get("critique", ""),
        sources=result.get("sources", []),
        latency_ms=total_latency_ms,
        node_latencies=node_latencies,
        agent_log=result.get("agent_log", [])
    )

@app.post("/stream")
def stream(request: QueryRequest):
    """
    Streaming inference endpoint.
    Returns Server-Sent Events (SSE) — one chunk per node completion.
    Client receives updates progressively instead of waiting for full pipeline.
    Headers:
    - Cache-Control: no-cache — prevents proxy caching of the stream
    - X-Accel-Buffering: no — prevents nginx from buffering the stream
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        stream_pipeline(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/metrics")
def get_metrics():
    """
    Basic observability endpoint.
    Returns aggregate stats across all queries since server start.
    Will be replaced by MLflow dashboard in the next step.
    """
    total = metrics["total_queries"]
    avg_latency = (
        round(metrics["total_latency_ms"] / total, 2) if total > 0 else 0.0
    )
    grounding_rate = (
        round(metrics["grounded_count"] / total * 100, 1) if total > 0 else 0.0
    )

    return {
        "total_queries": total,
        "avg_latency_ms": avg_latency,
        "grounding_rate_pct": grounding_rate,
        "grounded_count": metrics["grounded_count"],
        "ungrounded_count": metrics["ungrounded_count"],
        "mlflow_ui": "run 'mlflow ui' then visit http://localhost:5000"
    }