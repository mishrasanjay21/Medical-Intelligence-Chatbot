import json

import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/vectorstore/medical.index"
METADATA_FILE = "data/vectorstore/chunks_metadata.json"


def load_vectorstore():
    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    print("FAISS index loaded!")
    print(f"Total vectors: {index.ntotal}")

    print("Loading metadata...")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks loaded: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, chunks, model


def retrieve_documents(query, top_k=5):
    index, chunks, model = load_vectorstore()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_id in zip(distances[0], indices[0]):
        if index_id == -1:
            continue

        results.append({
            "text": chunks[index_id]["text"],
            "metadata": chunks[index_id]["metadata"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":
    query = "What are the symptoms of diabetes?"

    print("\nQuery:")
    print(query)

    results = retrieve_documents(query, top_k=5)

    print("\nRetrieved Documents:\n")

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Distance: {result['distance']}")
        print()
        print(result["text"])
        print()
