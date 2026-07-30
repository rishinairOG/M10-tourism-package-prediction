import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

def train_model():
    # Load train and test splits
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Preprocessing: scale numerical, one-hot encode categorical
    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        remainder='passthrough'
    )

    # Define model
    xgb = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Create pipeline
    pipeline = make_pipeline(preprocessor, xgb)

    # Define hyperparameter grid
    param_grid = {
        'xgbclassifier__n_estimators': [100, 200],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__learning_rate': [0.01, 0.1],
        'xgbclassifier__subsample': [0.8, 1.0]
    }

    # Start MLflow experiment
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run(run_name="XGBoost_GridSearch"):

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

        # Log all tuned parameters to MLflow
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Predictions on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nTest Set Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the best model to MLflow
        mlflow.sklearn.log_model(best_model, "xgboost_model")

    # Save the best model for deployment
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")
    print("\nModel saved to tourism_project/deployment/best_model.pkl")

if __name__ == "__main__":
    train_model()
