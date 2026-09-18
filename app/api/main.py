import random
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.rag.qa import ask_medical_question

app = FastAPI(
    title="Medical Intelligence Chatbot API",
    description="Retrieval-augmented medical information assistant.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.pkl"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS
]


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    disclaimer: str


class CustomerData(BaseModel):
    gender: str = Field(default="Male")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="Yes")
    Dependents: str = Field(default="No")
    tenure: int = Field(default=12, ge=0)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="DSL")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="No")
    StreamingMovies: str = Field(default="No")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=70.5, ge=0)
    TotalCharges: float = Field(default=850.0, ge=0)


def _build_training_dataframe():
    rows = []
    rng = random.Random(42)
    for idx in range(500):
        gender = "Male" if idx % 2 == 0 else "Female"
        senior = 1 if idx % 5 == 0 else 0
        partner = "Yes" if idx % 3 != 0 else "No"
        dependents = "Yes" if idx % 4 == 0 else "No"
        tenure = min(max(1, idx % 72 + 1), 72)
        phone_service = "Yes" if idx % 2 == 0 else "No"
        multiple_lines = "No" if idx % 4 == 0 else ("Yes" if idx % 3 == 0 else "No phone service")
        internet_service = ["DSL", "Fiber optic", "No"][idx % 3]
        online_security = ["No", "Yes", "No internet service"][idx % 3]
        online_backup = ["No", "Yes", "No internet service"][idx % 2]
        device_protection = ["No", "Yes", "No internet service"][idx % 3]
        tech_support = ["No", "Yes", "No internet service"][idx % 3]
        streaming_tv = ["No", "Yes", "No internet service"][idx % 3]
        streaming_movies = ["No", "Yes", "No internet service"][idx % 2]
        contract = ["Month-to-month", "One year", "Two year"][idx % 3]
        paperless = "Yes" if idx % 2 == 0 else "No"
        payment_method = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"][idx % 4]
        monthly = round(35 + (idx % 80) + (5 if internet_service == "Fiber optic" else 0), 2)
        total = round((tenure * monthly) + (50 if paperless == "Yes" else 0), 2)

        risk_score = 0.0
        risk_score += 0.25 if gender == "Female" else 0.08
        risk_score += 0.22 if senior == 1 else 0.0
        risk_score += 0.18 if partner == "No" else 0.0
        risk_score += 0.15 if dependents == "No" else 0.0
        risk_score += 0.12 if tenure < 12 else 0.0
        risk_score += 0.18 if phone_service == "No" else 0.0
        risk_score += 0.12 if multiple_lines == "No phone service" else 0.0
        risk_score += 0.18 if internet_service == "Fiber optic" else 0.05
        risk_score += 0.12 if online_security == "No" else 0.0
        risk_score += 0.10 if online_backup == "No" else 0.0
        risk_score += 0.09 if device_protection == "No" else 0.0
        risk_score += 0.10 if tech_support == "No" else 0.0
        risk_score += 0.07 if streaming_tv == "Yes" else 0.0
        risk_score += 0.08 if streaming_movies == "Yes" else 0.0
        risk_score += 0.20 if contract == "Month-to-month" else 0.08 if contract == "One year" else 0.0
        risk_score += 0.18 if paperless == "Yes" else 0.0
        risk_score += 0.22 if payment_method == "Electronic check" else 0.0
        risk_score += min(0.35, monthly / 200)
        risk_score += min(0.25, total / 3000)

        churn = 1 if idx % 2 == 0 else 0

        rows.append(
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "Churn": churn,
            }
        )

    return pd.DataFrame(rows)


def _load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    training_data = _build_training_dataframe()
    X = training_data[FEATURE_COLUMNS]
    y = training_data["Churn"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def _make_prediction_row(data: CustomerData) -> pd.DataFrame:
    payload = data.model_dump()
    return pd.DataFrame([payload], columns=FEATURE_COLUMNS)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(request: CustomerData):
    try:
        model = _load_or_train_model()
        row = _make_prediction_row(request)
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


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
