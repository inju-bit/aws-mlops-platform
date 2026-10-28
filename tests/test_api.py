from fastapi.testclient import TestClient
from app import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    flower = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=flower)

    assert response.status_code == 200

    result = response.json()
    assert result["flower_name"] == "setosa"
    assert result["predicted_class"] == 0
    assert 0 <= result["confidence"] <= 1


def test_invalid_input():
    flower = {
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=flower)

    assert response.status_code == 422
