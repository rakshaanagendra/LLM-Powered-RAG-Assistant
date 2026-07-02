from datetime import datetime

import faiss
import numpy as np
import pickle
import os
import sys
import json


def _normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)


def build_faiss_index(embeddings, chunks):
    # Step 1: Convert embeddings to numpy array
    embeddings = np.array(embeddings).astype("float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    embeddings = _normalize_embeddings(embeddings)

    # Step 2: Get vector dimension
    dim = embeddings.shape[1]

    if len(set(vec.shape[0] for vec in embeddings)) > 1:
        raise ValueError(
            "Embedding dimensions are inconsistent."
        )

    # Step 3: Create FAISS index for cosine similarity via inner product
    index = faiss.IndexFlatIP(dim)

    # Step 4: Add embeddings to index
    index.add(embeddings)  # pyright: ignore[reportCallIssue]

    # Step 5: Create mapping (VERY IMPORTANT)
    id_to_chunk = {i: chunks[i] for i in range(len(chunks))}

    return index, id_to_chunk


def save_index(index, mapping, save_path="data/processed"):
    os.makedirs(save_path, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(save_path, "faiss_index.bin"))

    metadata = {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": index.d,
    "num_chunks": len(mapping),
    "created_At": datetime.now().isoformat()
    }

    with open(
        os.path.join(save_path, "index_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    # Save mapping
    with open(os.path.join(save_path, "id_to_chunk.pkl"), "wb") as f:
        pickle.dump(mapping, f)


def load_index(load_path="data/processed"):
    """Load FAISS index and mapping from disk."""
    index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
    
    with open(os.path.join(load_path, "id_to_chunk.pkl"), "rb") as f:
        mapping = pickle.load(f)
    
    return index, mapping


# if __name__ == "__main__":
#     # For full pipeline, run from rag/ingestion first to generate embeddings
#     # This script assumes embeddings are already computed and saved
    
#     print("FAISS Indexer is ready.")
#     print("\nCore functions:")
#     print("  - build_faiss_index(embeddings, chunks) → (index, id_to_chunk)")
#     print("  - save_index(index, mapping, save_path)")
#     print("  - load_index(load_path) → (index, mapping)")

# if __name__ == "__main__":

#     PROJECT_ROOT = os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__),
#             ".."
#         )
#     )

#     processed_path = os.path.join(
#         PROJECT_ROOT,
#         "data",
#         "processed"
#     )

#     index, mapping = load_index(processed_path)

#     print(f"Total chunks: {len(mapping)}")

#     lora_chunks = []

#     for chunk in mapping.values():

#         source = chunk.get("source", "")

#         if "lora" in source.lower():

#             lora_chunks.append(chunk)

#     print(f"\nFound {len(lora_chunks)} LoRA chunks")

#     if lora_chunks:

#         print("\n" + "=" * 80)
#         print("FIRST LORA CHUNK")
#         print("=" * 80)

#         print(lora_chunks[0]["text"][:1500])

#         print("\n" + "=" * 80)