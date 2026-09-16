import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/medical_documents/train.csv"


def load_medical_data():
    return pd.read_csv(DATA_PATH)


def split_medical_documents(df):
    text_data = []

    for _, row in df.iterrows():
        text = (
            f"Question: {row['Question']}\n"
            f"Answer: {row['Answer']}"
        )
        text_data.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.create_documents(text_data)


if __name__ == "__main__":
    df = load_medical_data()
    chunks = split_medical_documents(df)

    print("Total chunks:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0].page_content)
