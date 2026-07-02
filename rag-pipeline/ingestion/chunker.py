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


def split_paragraphs(text):
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def merge_short_paragraphs(paragraphs, min_words=25):
    merged = []
    buffer = []
    buffer_word_count = 0

    for paragraph in paragraphs:
        word_count = len(re.findall(r"\b\w+\b", paragraph))

        if word_count < min_words:
            buffer.append(paragraph)
            buffer_word_count += word_count
            continue

        if buffer:
            merged.append("\n\n".join(buffer))
            buffer = []
            buffer_word_count = 0

        merged.append(paragraph)

    if buffer:
        merged.append("\n\n".join(buffer))

    return merged


def is_weak_chunk(chunk):
    text = chunk.strip()
    word_count = len(re.findall(r"\b\w+\b", text))
    alpha_count = sum(1 for char in text if char.isalpha())
    alpha_ratio = (alpha_count / len(text)) if text else 0.0

    if word_count < 8:
        return True

    if alpha_ratio < 0.35:
        return True

    lowered = text.lower()

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
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for doc_id, doc in enumerate(documents):
        cleaned_text = clean_text(doc["text"])
        paragraphs = split_paragraphs(cleaned_text)
        paragraph_blocks = merge_short_paragraphs(paragraphs)

        chunk_id = 0

        for block_index, block in enumerate(paragraph_blocks, start=1):
            split_texts = splitter.split_text(block)

            for chunk in split_texts:
                if is_weak_chunk(chunk):
                    continue

                chunk_data = {
                    "title": doc.get("title"),
                    "text": chunk,
                    "word_count": len(
                        re.findall(r"\b\w+\b", chunk)
                    ),
                    "source": doc.get("source", "unknown"),
                    "source_path": doc.get("source_path", ""),
                    "category": doc.get("category"),
                    "doc_type": doc.get("doc_type", "unknown"),
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "length": len(chunk),
                    "block_id": block_index,
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

    target_source = "REACT"
    source_chunks = [c for c in chunks if target_source.lower() in c.get("source", "").lower()]

    print(f"\nChunks for source containing '{target_source}' ({len(source_chunks)}):")
    for i, chunk in enumerate(source_chunks[:5], start=1):
        print("\n==================================================")
        print(f"Chunk {i}/{len(source_chunks)}")
        print("==================================================")
        print(json.dumps(chunk, ensure_ascii=False, indent=2))

