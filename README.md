# AWS MLOps Model Deployment Platform

An end-to-end Machine Learning Operations (MLOps) project demonstrating model training, experiment tracking, REST API development, containerisation, automated testing, and CI/CD using Python, MLflow, FastAPI, Docker, GitHub Actions, and AWS.

The project uses the Iris dataset to train a Random Forest classification model, tracks experiments using MLflow, and exposes predictions through a FastAPI service.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | NumPy |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib, MLflow Skops |
| Backend API | FastAPI, Uvicorn |
| Containerisation | Docker |
| Automated Testing | Pytest, HTTPX |
| CI/CD | GitHub Actions |
| Cloud Deployment (Planned) | AWS ECR, AWS App Runner |
| Version Control | Git, GitHub |

## Project Architecture

```text
Iris Dataset
     |
     v
Data Preprocessing
     |
     v
Train/Test Split
     |
     v
Random Forest Training
     |
     v
Model Evaluation
     |
     +------> MLflow Experiment Tracking
     |
     v
Model Serialization
     |
     v
FastAPI Prediction Service
     |
     v
Docker Container
     |
     v
AWS ECR + App Runner (Planned)
```

GitHub Actions provides a separate CI pipeline for training, testing, and building the Docker image.

## Project Structure

```text
aws-mlops-platform/
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|
|-- tests/
|   |-- test_api.py
|
|-- models/
|   |-- iris_model.pkl  # Generated locally
|
|-- app.py
|-- train.py
|-- Dockerfile
|-- requirements.txt
|-- .gitignore
|-- README.md
```

The trained model, MLflow database, and virtual environment are generated locally and excluded from Git version control.

## Key Features

### 1. Machine Learning Training Pipeline

- Loads the Iris classification dataset.
- Splits data into training and testing sets.
- Trains a Random Forest classifier.
- Evaluates model performance using accuracy and classification metrics.
- Saves the trained model for inference.

### 2. MLflow Experiment Tracking

- Records training parameters.
- Tracks model accuracy.
- Stores model artifacts.
- Maintains experiment history.
- Supports comparing different training runs.

### 3. FastAPI Prediction Service

Provides an HTTP-based prediction API with automatic Swagger documentation.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Model health status |
| `/predict` | POST | Predict Iris flower species |
| `/docs` | GET | Interactive API documentation |

### 4. Docker Containerisation

Packages the FastAPI application, Python dependencies, and trained model into a portable Docker image.

### 5. Automated Testing

Uses Pytest to validate:

- API home endpoint.
- Model health endpoint.
- Successful predictions.
- Invalid input handling.

### 6. GitHub Actions CI Pipeline

The workflow is configured to:

1. Check out repository code.
2. Set up Python 3.12.
3. Install dependencies.
4. Train the machine-learning model.
5. Execute automated tests.
6. Build the Docker image.

The workflow runs on pushes and pull requests to the `main` branch. Successful CI execution should be verified in the GitHub Actions tab.

## Installation and Local Setup

### Prerequisites

- Python 3.12
- Git
- Docker Desktop
- VS Code or another code editor

### Clone the repository

```bash
git clone https://github.com/inju-bit/aws-mlops-platform.git
cd aws-mlops-platform
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install mlflow skops pytest httpx
```

## Train the Model

Run the training pipeline:

```bash
python train.py
```

The pipeline trains a Random Forest model, evaluates it, saves the trained model in the `models/` directory, and records experiment results with MLflow.

## View MLflow Experiments

Start the MLflow tracking interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

http://127.0.0.1:5000

The dashboard displays experiment runs, parameters, metrics, and artifacts.

## Run FastAPI Locally

Start the prediction server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8005
```

Open the interactive API documentation:

http://127.0.0.1:8005/docs

## Example Prediction

**Endpoint:**

```text
POST /predict
```

**Request:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Example response from local testing:**

```json
{
  "predicted_class": 0,
  "flower_name": "setosa",
  "confidence": 1,
  "probabilities": {
    "setosa": 1,
    "versicolor": 0,
    "virginica": 0
  }
}
```

Prediction confidence depends on the trained model and should not be interpreted as overall model accuracy.

## Docker Deployment

### Build the Docker image

First, train the model locally so that the model artifact exists.

```bash
python train.py
docker build -t aws-mlops-api:v1 .
```

### Run the container

```bash
docker run -d \
  --name aws-mlops-container \
  -p 8006:8000 \
  aws-mlops-api:v1
```

### Verify the container

```bash
docker ps
```

Open:

http://127.0.0.1:8006/docs

## Automated Testing

Run the test suite:

```bash
python -m pytest -v tests/
```

The tests validate HTTP responses, model availability, predictions, and request validation.

## Cloud Deployment Roadmap

The planned cloud architecture uses:

- **Amazon ECR:** Stores Docker container images.
- **AWS App Runner:** Hosts the containerised prediction API.
- **GitHub Actions:** Automates quality checks and image builds.
- **AWS IAM:** Controls deployment permissions.

Cloud deployment is not yet implemented or verified. The existing GitHub Actions workflow performs CI tasks but does not currently publish images to ECR or deploy to App Runner.

## Future Improvements

- Add automated model version management and promotion.
- Implement production model monitoring and data drift detection.
- Add input and prediction logging.
- Create an AWS deployment workflow with secure IAM authentication.
- Implement automatic model retraining.
- Add infrastructure as code using Terraform.
- Expand the testing suite with model quality and integration tests.

## Skills Demonstrated

- Machine-learning pipeline development.
- Supervised classification and model evaluation.
- MLflow experiment tracking.
- REST API development with FastAPI.
- Docker containerisation.
- Automated software testing.
- GitHub Actions CI/CD configuration.
- Cloud deployment architecture and MLOps fundamentals.

## Project Status

Local model training and FastAPI inference have been tested successfully. Docker, test automation, GitHub Actions, and AWS deployment require final end-to-end verification.

## Author

**GitHub:** [inju-bit](https://github.com/inju-bit)

## License

No licence has been assigned yet.
