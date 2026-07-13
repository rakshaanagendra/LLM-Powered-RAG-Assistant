import os
from dotenv import load_dotenv
load_dotenv()

import httpx
from fastmcp import FastMCP

# -----------------------------------------------------------------------
# MCP Server
# Why: FastMCP creates an MCP-compliant server that exposes your RAG
# pipeline as a callable tool. Any MCP client (Claude Desktop, Cursor,
# etc.) can discover and call this tool without custom integration code.
# -----------------------------------------------------------------------
mcp = FastMCP(
    name="production-rag-agent")

# -----------------------------------------------------------------------
# FastAPI base URL
# Why: MCP server is a thin wrapper — it delegates all actual work to
# your existing FastAPI endpoint. No pipeline logic lives here.
# Change this URL when deploying to production.
# -----------------------------------------------------------------------
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

# -----------------------------------------------------------------------
# Tool: query_rag_pipeline
# Why: this is what the MCP client sees and calls. The description is
# critical — the AI reads it to decide when to use this tool.
# Rule: description must clearly state what topics this tool covers
# so the AI calls it for the right queries.
# -----------------------------------------------------------------------
@mcp.tool()
async def query_rag_pipeline(question: str) -> str:
    """
    Query a production RAG multi-agent system specialising in AI research papers.
    
    Use this tool for questions about:
    - Retrieval-Augmented Generation (RAG) and its variants
    - LLM engineering and prompt engineering
    - Agentic AI, ReAct agents, and multi-agent systems
    - LangGraph, LangChain, and related frameworks
    - LLMOps, evaluation, and deployment of LLM systems
    
    Input: a natural language question about any of the above topics.
    Output: a grounded answer with source citations from research papers.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/query",
                json={"question": question}
            )
            response.raise_for_status()
            data = response.json()

            # Format the response clearly for the MCP client
            answer = data.get("answer", "No answer generated.")
            sources = data.get("sources", [])
            confidence = data.get("confidence", "unknown")
            grounded = data.get("grounded", False)
            latency_ms = data.get("latency_ms", 0)

            return (
                f"{answer}\n\n"
                f"---\n"
                f"Sources: {', '.join(sources) if sources else 'None'}\n"
                f"Confidence: {confidence} | "
                f"Grounded: {grounded} | "
                f"Latency: {latency_ms}ms"
            )

        except httpx.TimeoutException:
            return "Error: RAG pipeline timed out after 120 seconds."
        except httpx.HTTPStatusError as e:
            return f"Error: FastAPI returned {e.response.status_code} — {e.response.text}"
        except Exception as e:
            return f"Error: {str(e)}"


# -----------------------------------------------------------------------
# Run the MCP server
# transport="streamable-http" — runs as HTTP server, not subprocess
# Why streamable-http not stdio: your FastAPI is already an HTTP service;
# stdio is for local subprocess tools, not network-accessible APIs
# -----------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8001,        # different port from FastAPI (8000)
    )