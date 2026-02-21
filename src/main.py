from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict

app = FastAPI()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Request body schema (like Joi validation)
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    result = predict(request.text)

    # Handle error from model
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "model": MODEL_NAME
    }