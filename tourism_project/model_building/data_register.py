import pandas as pd
import sys
import os

def validate_dataset(filepath):
    """
    Validates the tourism dataset by checking expected columns
    and printing a summary.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)

    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]

    missing_cols = set(expected_columns) - set(df.columns)

    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        sys.exit(1)

    print("All expected columns are present.")
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nTarget variable (ProdTaken) distribution:")
    print(df['ProdTaken'].value_counts().to_string())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_only = missing[missing > 0]
    if len(missing_only) > 0:
        print(missing_only.to_string())
    else:
        print("None")
    print(f"\nData types:")
    print(df.dtypes.to_string())

if __name__ == "__main__":
    validate_dataset("tourism_project/data/tourism.csv")
