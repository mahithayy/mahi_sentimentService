from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.model import predict

app = FastAPI()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    result = predict(request.text)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "model": MODEL_NAME
    }


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []

    for text in request.texts:
        result = predict(text)

        if "error" in result:
            results.append({"error": result["error"]})
        else:
            results.append({
                "label": result["label"],
                "confidence": result["confidence"]
            })

    return {
        "model": MODEL_NAME,
        "results": results
    }