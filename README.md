# Sentiment Service & Career Graph API

*By N Mahitha Holla — 22/02/2026*

## Overview

This project implements:

### 1. Sentiment Analysis Microservice
- Loads a HuggingFace sentiment model
- Exposes REST endpoints for predictions
- Handles validation and edge cases
- Includes pytest-based tests

### 2. Career Transition Graph Builder
- Builds a PyTorch Geometric graph from career transitions
- Maps job titles to indices
- Supports one-hop neighbor lookup

This project mirrors real-world ML service deployment and graph-based data modeling.

---
## Requirements
- Python 3.10+
## Features

### Sentiment API
- POST `/predict` — single text prediction
- POST `/predict/batch` — batch predictions (bonus)
- Input validation with proper HTTP error codes
- Model loads once at startup for efficiency

### Graph Module
- Builds `torch_geometric.data.Data` graph
- Dynamic job → index mapping
- Edge attributes store transition years
- One-hop neighbor lookup

### Testing
- Happy path
- Edge cases
- Error handling

---

## Tech Stack

- **FastAPI** — REST API
- **HuggingFace Transformers** — sentiment model
- **PyTorch & PyTorch Geometric** — graph construction
- **pytest** — testing
- **Pydantic** — request validation

---

## Model Used

**distilbert-base-uncased-finetuned-sst-2-english**

### Why this model?
- Small and fast (suitable for microservices)
- High accuracy for sentiment analysis
- No GPU required
- Ideal for low-latency inference

---

## Project Structure

```

sentiment-service/
│
├── src/
│   ├── main.py        # FastAPI app & endpoints
│   ├── model.py       # Model loading & prediction
│   └── graph.py       # Graph construction logic
│
├── tests/
│   ├── test_api.py
│   └── test_graph.py
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Setup Instructions (From Scratch)

```bash
git clone <your-repo-url>
cd sentiment-service
python -m venv venv
```

### Activate Virtual Environment

**Windows (PowerShell)**

```bash
venv\Scripts\activate
```

**Git Bash**

```bash
source venv/Scripts/activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
uvicorn src.main:app --reload
```

Open Swagger docs:
http://127.0.0.1:8000/docs
---

## API Usage

### POST /predict

```json
{
  "text": "I love this product"
}
```

---

## Edge Case Handling

### Sentiment API

* Empty input → HTTP 422
* Missing field → HTTP 422
* Very long input → truncated to 512 characters
* Model loads once at startup for performance

### Graph Builder

* Empty transitions → returns empty graph
* Self-loops skipped (design choice)
* No edges → returns graph with mapping only

---

## Running Tests

```bash
export PYTHONPATH=.
pytest
```
### Windows PowerShell

```bash
set PYTHONPATH=.
pytest
### Expected Output

```
tests/test_api.py ..... PASSED
tests/test_graph.py ... PASSED
```
