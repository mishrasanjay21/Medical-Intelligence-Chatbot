from app.llm.groq_llm import create_llm
from app.rag.retriever import retrieve_documents


def ask_medical_question(question, top_k=5):
    results = retrieve_documents(question, top_k=top_k)

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
You are a medical information assistant.

Answer the user's question using only the information
provided in the context below.

If the answer is not available in the context,
clearly say that the information is not available
in the provided medical documents.

Do not invent medical facts.

Context:
{context}

User Question:
{question}

Answer:
"""

    llm = create_llm()
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    question = "What are the symptoms of diabetes?"

    print("\nUser Question:")
    print(question)

    print("\nGenerating answer...")
    answer = ask_medical_question(question)

    print("\nFinal Answer:")
    print(answer)
