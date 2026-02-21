from transformers import pipeline

# Load model once at startup (like DB connection)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict(text: str) -> dict:
    """
    Predict sentiment for a given text.

    Returns:
        dict: { label: str, confidence: float }
    """

    # Edge case: empty input
    if not text or text.strip() == "":
        return {"error": "Input text cannot be empty"}

    # Handle very long input (truncate)
    text = text[:512]

    result = sentiment_pipeline(text)[0]

    return {
        "label": result["label"],
        "confidence": float(result["score"])
    }