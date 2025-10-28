from typing import Literal

import pickle

from fastapi import FastAPI
from pydantic import BaseModel


class Lead(BaseModel):
    lead_source: Literal[
        "paid_ads",
        "organic_search",
        "referral",
        "social_media",
        "events",
        "email_marketing",
        "direct",
    ]
    number_of_courses_viewed: int
    annual_income: float


with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict")
def predict(lead: Lead):
    proba = model.predict_proba([lead.model_dump()])[0, 1]
    return {"converted_probability": proba}
