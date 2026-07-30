import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(filepath):
    """
    Loads the dataset, cleans it, and splits into train/test sets.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')
    print(f"After dropping unnecessary columns: {df.shape[0]} rows x {df.shape[1]} columns")

    # Handle missing values - fill numerical with median, categorical with mode
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled {col} missing values with mode: {mode_val}")

    print(f"\nRemaining missing values: {df.isnull().sum().sum()}")

    # Separate features and target
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    # Split into train and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")
    print(f"Target distribution in train set:\n{y_train.value_counts().to_string()}")
    print(f"Target distribution in test set:\n{y_test.value_counts().to_string()}")

    # Save splits as CSV
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("\nSaved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

if __name__ == "__main__":
    prepare_data("tourism_project/data/tourism.csv")
