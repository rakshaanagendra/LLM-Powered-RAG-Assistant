from pydoc import doc

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
import json
import re

try:
    from .loader import load_documents
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from loader import load_documents  # pyright: ignore


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.split(r"(?i)\breferences\b", text)[0]
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def is_weak_chunk(chunk):
    text = chunk.strip().lower()

    if len(text) < 100:
        return True

    weak_patterns = [
        r"^page\s+\d+$",
        r"^\d+$",
        r"^references?$",
        r"^table of contents$",
        r"^figure\s+\d+.*$",
        r"^table\s+\d+.*$",
    ]

    for pattern in weak_patterns:
        if re.match(pattern, text):
            return True

    return False


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for doc_id, doc in enumerate(documents):
        cleaned_text = clean_text(doc["text"])

        split_texts = splitter.split_text(cleaned_text)
        chunk_id = 0

        for chunk in split_texts:
            if is_weak_chunk(chunk):
                continue

            chunk_data = {
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "source_path": doc.get("source_path", ""),
                "doc_type": doc.get("doc_type", "unknown"),
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "length": len(chunk),
            }

            chunk_data["page"] = doc.get("page")
            chunk_data["title"] = doc.get("title")

            chunks.append(chunk_data)
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resolved_data_path = os.path.join(project_root, "data", "raw")

    documents = load_documents(resolved_data_path)
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} document(s)")
    print(f"Created {len(chunks)} chunk(s)")

    print("\nSample chunks:")
    for chunk in chunks[:3]:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))