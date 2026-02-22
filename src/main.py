from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from src.model import predict
import time
import logging

app = FastAPI()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


# 🔹 Request timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} completed in {duration:.4f}s")

    return response


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    result = predict(request.text)

    # 🔹 Log prediction result
    logger.info(f"Prediction made: {result}")

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