import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# STEP 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "students.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create outputs folder if it does not exist
OUTPUT_DIR.mkdir(exist_ok=True)

# STEP 2: Load dataset
df = pd.read_csv(DATA_PATH)

# STEP 3: Display first five rows
print("\n1. FIRST FIVE ROWS")
print(df.head())

# STEP 4: Dataset shape
print("\n2. DATASET SHAPE")
print(df.shape)

# STEP 5: Dataset information
print("\n3. DATASET INFORMATION")
df.info()

# STEP 6: Check missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# STEP 7: Check duplicates
print("\n5. DUPLICATE RECORDS")
print(df.duplicated().sum())

# STEP 8: Statistical summary
print("\n6. STATISTICAL SUMMARY")
print(df.describe())


# STEP 9: Correlation analysis
print("\n7. CORRELATION")
print(df.corr(numeric_only=True))


# STEP 10: Visualise study hours vs final marks
plt.figure(figsize=(8, 5))

plt.scatter(
    df["study_hours"],
    df["final_marks"],
    color="blue"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "study_hours_vs_marks.png"
)

plt.close()


# STEP 11: Final marks distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["final_marks"],
    bins=10,
    color="green",
    edgecolor="black"
)

plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Marks")

plt.savefig(
    OUTPUT_DIR / "marks_distribution.png"
)

plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in outputs folder.")
