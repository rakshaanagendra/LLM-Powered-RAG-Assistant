#### LLM-Powered RAG Assistant

# Quick Start

## Clone Repository

```bash
git clone https://github.com/rakshaanagendra/llm-rag-assistant.git
cd llm-rag-assistant
```

## Create Virtual Environment

```bash
python -m venv rag
```

## Activate Virtual Environment

### Windows
```bash
.\rag\Scripts\activate
```

### Linux / Mac
```bash
source rag/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python main.py
```

---

# Example Workflow

## Example Query

```text
What is the difference between dense and sparse retrieval?
```

## Example Pipeline Flow

```text
Query
  ↓
Multi-Query Expansion
  ↓
Dense + Sparse Retrieval
  ↓
Hybrid Retrieval (RRF)
  ↓
Cross-Encoder Reranking
  ↓
Grounded Context Construction
  ↓
LLM Response Generation
  ↓
Semantic Validation + Confidence Scoring
```

## Example Output

```text
Dense retrieval uses semantic embeddings to retrieve contextually similar documents, while sparse retrieval relies on keyword matching techniques such as BM25. Hybrid retrieval combines both approaches to improve retrieval robustness and grounding quality.
```

---

# Repository Structure

```text
project/
│
├── data/                  # Raw and processed knowledge files
├── embeddings/            # Embedding generation and vector indexing
├── retrieval/             # Dense, sparse, hybrid retrieval and RRF logic
├── reranking/             # Cross-Encoder reranking pipelines
├── evaluation/            # Validation, confidence scoring and evaluation workflows
├── generation/            # Prompt construction and grounded generation
├── validation/            # Semantic validation and hallucination mitigation
├── app/                   # Application/API layer
├── notebooks/             # Experimental notebooks and testing
├── requirements.txt
└── README.md
```

---

# Project Status

## Completed
- Dense vector retrieval
- BM25 sparse retrieval
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Multi-query retrieval expansion
- Cross-Encoder reranking
- Grounded response generation
- Citation-aware response workflows
- Semantic validation
- LLM-as-a-judge evaluation
- Confidence scoring
- Safe abstention handling

## In Progress
- Retrieval diagnostics
- Metadata-aware retrieval
- Dynamic top-k retrieval
- Context compression
- Query routing/classification

## Planned
- UI-based AI assistant
- End-to-end deployment pipeline
- Observability and tracing
- Caching and latency optimization
- Advanced evaluation dashboards
- Scalable cloud deployment
- Agentic retrieval workflows

---

# Why This Project?

Most beginner RAG systems stop at:
- vector search
- prompt injection
- response generation

This project focuses on the engineering challenges behind reliable LLM systems:
- retrieval quality
- reranking effectiveness
- grounded generation
- hallucination mitigation
- evaluation workflows
- modular AI system design

The goal is to understand and implement production-oriented RAG architectures rather than building a basic chatbot wrapper.