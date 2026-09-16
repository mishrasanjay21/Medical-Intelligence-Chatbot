import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/processed/chunks.json"
VECTORSTORE_DIR = "data/vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "medical.index")
METADATA_FILE = os.path.join(VECTORSTORE_DIR, "chunks_metadata.json")


def create_vectorstore():
    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks loaded: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded successfully!")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    print("This may take some time...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    print("Embeddings created successfully!")
    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("FAISS index created successfully!")
    print(f"Total vectors in index: {index.ntotal}")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("\nVector store saved successfully!")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    create_vectorstore()
