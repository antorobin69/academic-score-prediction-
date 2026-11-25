
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# STEP 1: Load the dataset
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "students.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# STEP 2: Separate input features and target
features = [
    "study_hours",
    "attendance",
    "previous_marks"
]

X = df[features]

y = df["final_marks"]

print("\nInput features:")
print(X.head())

print("\nTarget:")
print(y.head())

# STEP 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 4: Check missing values before preparation
print("\nMissing values before preprocessing:")
print(X_train.isnull().sum())

# STEP 5: Build preprocessing pipeline
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# STEP 6: Prepare the training data
X_train_prepared = preprocessor.fit_transform(X_train)

# STEP 7: Prepare test data using training parameters
X_test_prepared = preprocessor.transform(X_test)

# STEP 8: Convert prepared arrays to DataFrames
X_train_clean = pd.DataFrame(
    X_train_prepared,
    columns=features,
    index=X_train.index
)

X_test_clean = pd.DataFrame(
    X_test_prepared,
    columns=features,
    index=X_test.index
)

# STEP 9: Check results
print("\nMissing values after preparation:")
print(X_train_clean.isnull().sum())

print("\nPrepared training data:")
print(X_train_clean.head())

print("\nPrepared testing data:")
print(X_test_clean.head())

print("\nTraining shape:", X_train_clean.shape)
print("Testing shape:", X_test_clean.shape)

print("\nDATA PREPARATION COMPLETED!")