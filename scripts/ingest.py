import sys
from pathlib import Path

# Allow running this file directly: `python scripts/ingest.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_documents
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.embedder import embed_chunks
from vectorstore.faiss_indexer import build_faiss_index, save_index


def run_ingestion():
    data_path = PROJECT_ROOT / "data" / "raw"
    processed_path = PROJECT_ROOT / "data" / "processed"

    print("Loading documents...")
    docs = load_documents(str(data_path))
    if not docs:
        print(f"No documents found in: {data_path}")
        print("Add .txt or .pdf files under data/raw and run again.")
        return

    print("Chunking...")
    chunks = chunk_documents(docs)
    if not chunks:
        print("No chunks were generated from the loaded documents.")
        return

    print("Embedding...")
    embeddings = embed_chunks(chunks)
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings were generated.")
        return

    print("Building FAISS index...")
    index, mapping = build_faiss_index(embeddings, chunks)

    print("Saving index...")
    save_index(index, mapping, str(processed_path))

    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()