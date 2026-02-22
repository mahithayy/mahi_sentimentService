# Sentiment Service & Career Graph API

## Overview

This project implements a sentiment analysis microservice and a career transition graph builder using Python, FastAPI, HuggingFace Transformers, and PyTorch Geometric.

## Features

* Sentiment prediction API using a pre-trained HuggingFace model
* Input validation and error handling
* Automated tests with pytest
* Career transition graph construction
* Neighbor lookup for graph nodes

## Tech Stack

* FastAPI
* HuggingFace Transformers
* PyTorch & PyTorch Geometric
* pytest

## Project Structure

```
src/
  main.py
  model.py
  graph.py
tests/
  test_api.py
  test_graph.py
```

## Setup Instructions

```bash
git clone <repo-url>
cd sentiment-service
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Run API

```bash
uvicorn src.main:app --reload
```

Visit: http://127.0.0.1:8000/docs

## Run Tests

```bash
export PYTHONPATH=src
pytest
```

## Model Used

distilbert-base-uncased-finetuned-sst-2-english
Chosen for its small size and high accuracy for sentiment analysis.
## Edge Case Handling

* Empty input returns HTTP 422 with a validation error.
* Very long input is truncated to 512 characters to ensure efficient model inference.
