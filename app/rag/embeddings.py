from sentence_transformers import SentenceTransformer


def create_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embeddings model loaded successfully!")

    return model


if __name__ == "__main__":
    create_embeddings()
