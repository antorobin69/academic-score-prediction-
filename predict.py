
# Import required libraries
from pathlib import Path
from functools import lru_cache
import math

import joblib
import numpy as np
import pandas as pd


# STEP 1: Define the model path
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml_models" / "marks_model.joblib"


# STEP 2: Load the trained model
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run python train.py first."
        )

    model_bundle = joblib.load(MODEL_PATH)

    return model_bundle


# STEP 3: Create a prediction function
def predict_marks(study_hours, attendance, previous_marks):

    # Validate the inputs
    inputs = [
        ("Study hours", study_hours, 0, 12),
        ("Attendance", attendance, 0, 100),
        ("Previous marks", previous_marks, 0, 100)
    ]

    for name, value, minimum, maximum in inputs:

        if not math.isfinite(value):
            raise ValueError(f"{name} must be a valid number")

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}"
            )

    # Load the trained model
    bundle = load_model()

    # Convert new student input into a DataFrame
    student_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks
    }])

    # Maintain the correct input column order
    student_data = student_data[bundle["features"]]

    # Predict final marks
    predicted_marks = bundle["model"].predict(student_data)[0]

    # Keep marks between 0 and 100
    predicted_marks = np.clip(predicted_marks, 0, 100)

    return round(float(predicted_marks), 2)


# STEP 4: Test the prediction
if __name__ == "__main__":

    result = predict_marks(
        study_hours=5,
        attendance=85,
        previous_marks=70
    )

    print("Predicted Final Marks:", result)
