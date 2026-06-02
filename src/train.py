import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "models/linear_regression_pima.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    mlflow.set_experiment("Pima_Diabetes_Linear_Regression")

    with mlflow.start_run(run_name="linear_regression_pima"):

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Convert regression output to class prediction
        y_pred_class = [1 if pred >= 0.5 else 0 for pred in y_pred]
        accuracy = accuracy_score(y_test, y_pred_class)

        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("classification_accuracy_threshold_0.5", accuracy)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="linear_regression_model"
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        print("Training completed successfully.")
        print(f"MSE      : {mse:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"R2 Score : {r2:.4f}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
