import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = "StudentsPerformance.csv"
PASS_MARK = 50  # math score >= 50 -> PASS

CATEGORICAL_FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
NUMERIC_FEATURES = ["reading_score", "writing_score"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_dataset():
    data = pd.read_csv(DATASET_PATH)

    # "race/ethnicity" -> "race_ethnicity", "math score" -> "math_score", ...
    data.columns = [
        column.strip().lower().replace("/", "_").replace(" ", "_")
        for column in data.columns
    ]

    # 1 = PASS, 0 = FAIL (predict math result from the other information)
    data["result"] = (data["math_score"] >= PASS_MARK).astype(int)
    return data


def train_model():
    print("Loading dataset...")
    data = load_dataset()
    data.to_csv("student_performance_processed.csv", index=False)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("PASS records     :", int(data["result"].sum()))
    print("FAIL records     :", int((data["result"] == 0).sum()))

    X = data[FEATURES]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "student_performance_model.pkl")
    print("\nModel saved as student_performance_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
