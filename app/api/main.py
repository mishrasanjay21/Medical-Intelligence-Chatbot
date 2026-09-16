from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.qa import ask_medical_question

app = FastAPI(
    title="Medical Intelligence Chatbot API",
    description="Retrieval-augmented medical information assistant.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    disclaimer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = ask_medical_question(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer right now."
        ) from exc

    return AnswerResponse(
        question=request.question,
        answer=answer,
        disclaimer=(
            "This information is for educational purposes only and is not a "
            "substitute for advice from a qualified healthcare professional."
        )
    )
