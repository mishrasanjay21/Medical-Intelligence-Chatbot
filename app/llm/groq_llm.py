import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def create_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=api_key
    )


if __name__ == "__main__":
    print("Loading Groq LLM...")

    llm = create_llm()
    response = llm.invoke(
        "What are the common symptoms of diabetes?"
    )

    print("\nGroq Response:\n")
    print(response.content)
