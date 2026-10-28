
import os
import joblib
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create model directory
os.makedirs("models", exist_ok=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AWS-MLOps-Iris-Classification")

# Start experiment
with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log experiment information
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)

    # Save model locally
    model_path = "models/iris_model.pkl"
    joblib.dump(model, model_path)

    # Record input/output signature
    signature = infer_signature(X_test, predictions)

    # Save model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_classifier",
        signature=signature,
        serialization_format="skops",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print("\nTraining completed successfully!")
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
    )
    print(f"\nModel saved to: {model_path}")
    print("Model logged to MLflow successfully!")
