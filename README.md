# Wellness Tourism Package Prediction — MLOps Pipeline

An end-to-end MLOps pipeline that predicts whether a customer will purchase the newly introduced **Wellness Tourism Package** for the travel company "Visit with Us." The project automates the full machine learning workflow — from data registration through model training to deployment — using GitHub Actions for CI/CD and Streamlit Community Cloud for the live app.

## Problem Statement

"Visit with Us" faces challenges in efficiently targeting the right customers for its new Wellness Tourism Package. The manual approach to identifying potential buyers is inconsistent and time-consuming. This project builds a scalable, automated system that predicts potential buyers before they are contacted, enabling smarter marketing decisions and better customer acquisition.

## Project Structure

```
tourism_project/
├── data/
│   └── tourism.csv                 # Raw dataset
├── model_building/
│   ├── data_register.py            # Validates dataset columns and prints summary
│   ├── prep.py                     # Cleans data, splits into train/test sets
│   └── train.py                    # Tunes XGBoost, logs to MLflow, saves best model
├── deployment/
│   ├── app.py                      # Streamlit frontend for predictions
│   ├── best_model.pkl              # Trained model (committed by the pipeline)
│   └── requirements.txt            # Deployment dependencies
└── requirements.txt                # Pipeline dependencies

.github/workflows/
└── pipeline.yml                    # GitHub Actions CI/CD workflow
```

## Pipeline Workflow

The pipeline runs automatically on every push to the `main` branch and consists of three sequential jobs:

1. **register-dataset** — Validates the dataset schema and uploads it as a workflow artifact.
2. **data-prep** — Cleans the data, removes unnecessary columns, splits into train/test sets, and passes them to the next job as artifacts.
3. **model-training** — Trains and tunes an XGBoost classifier via GridSearchCV, logs parameters and metrics to MLflow, and commits the best model back to the repository.

Once the model is committed, the Streamlit app loads it to serve live predictions.

## Model Details

- **Algorithm:** XGBoost Classifier
- **Preprocessing:** StandardScaler for numerical features, OneHotEncoder for categorical features
- **Hyperparameter tuning:** GridSearchCV with 3-fold cross-validation, scoring on F1 (due to class imbalance)
- **Experiment tracking:** MLflow

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 93.2% |
| Precision | 92.6% |
| Recall | 70.4% |
| F1 Score | 80.0% |
| ROC AUC | 96.4% |

## Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **CI/CD:** GitHub Actions
- **Deployment:** Streamlit Community Cloud

## Links

- **GitHub Repository:** https://github.com/rishinairOG/M10-tourism-package-prediction
- **Live App:** https://m10-tourism-package-prediction-8qwckvk4ccznagheiycnfc.streamlit.app

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/rishinairOG/M10-tourism-package-prediction.git
cd M10-tourism-package-prediction

# Install dependencies
pip install -r tourism_project/deployment/requirements.txt

# Run the Streamlit app
streamlit run tourism_project/deployment/app.py
```
