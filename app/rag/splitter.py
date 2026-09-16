from app.rag.loader import load_medical_data, split_medical_documents
import json
import os


def save_chunks(chunks):
    # Create processed folder
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Convert LangChain Documents into JSON format
    chunks_data = []

    for chunk in chunks:
        chunks_data.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    # Output file
    output_file = os.path.join(output_dir, "chunks.json")

    # Save chunks
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            chunks_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Chunks saved successfully to: {output_file}")


if __name__ == "__main__":

    # Step 1: Load medical dataset
    print("Loading medical dataset...")
    df = load_medical_data()

    # Step 2: Split documents into chunks
    print("Creating chunks...")
    chunks = split_medical_documents(df)

    # Step 3: Show total chunks
    print("Total chunks:", len(chunks))

    # Step 4: Save chunks
    save_chunks(chunks)

    # Step 5: Show first chunk
    if chunks:
        print("\nFirst chunk:\n")
        print(chunks[0].page_content)
    else:
        print("No chunks were created.")