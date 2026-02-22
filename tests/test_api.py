from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_happy_path():
    response = client.post("/predict", json={"text": "I love this"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data

def test_empty_input():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422