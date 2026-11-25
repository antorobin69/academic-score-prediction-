
# Import libraries
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# STEP 1: Define file paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "students.csv"

MODEL_DIR = BASE_DIR / "ml_models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# STEP 2: Load the CSV dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# STEP 3: Separate input features and target
FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks"
]

X = df[FEATURES]
y = df["final_marks"]


# STEP 4: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 5: Build the ML model and pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])


# STEP 6: Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# STEP 7: Predict using test data
y_pred = model.predict(X_test)

# Keep predicted marks between 0 and 100
y_pred = np.clip(y_pred, 0, 100)


# STEP 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))


# STEP 9: Compare actual and predicted marks
results = X_test.copy()

results["actual_marks"] = y_test.to_numpy()
results["predicted_marks"] = np.round(y_pred, 2)

print("\nACTUAL VS PREDICTED MARKS")
print(results.head(10))

# Save comparison as a CSV
results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# STEP 10: Save the trained ML model
MODEL_PATH = MODEL_DIR / "marks_model.joblib"

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)

print("\nTrained model saved successfully!")
print("Saved at:", MODEL_PATH)
