from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("student_performance_model.pkl")
FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score",
]


OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Student Performance Prediction API",
        "version": "1.0.0",
        "description": (
            "Predicts whether a student will PASS or FAIL mathematics "
            "(math score >= 50) using the StudentsPerformance dataset."
        ),
    },
    "paths": {
        "/": {
            "get": {
                "summary": "Service status",
                "tags": ["Health"],
                "responses": {
                    "200": {
                        "description": "Service is running",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/Status"}}},
                    }
                },
            }
        },
        "/health": {
            "get": {
                "summary": "Health check",
                "tags": ["Health"],
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/Status"}}},
                    }
                },
            }
        },
        "/predict": {
            "post": {
                "summary": "Predict student math result (PASS / FAIL)",
                "tags": ["Prediction"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/PredictRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Prediction result",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/PredictResponse"}}},
                    },
                    "400": {
                        "description": "Missing or invalid request body",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/Error"}}},
                    },
                },
            }
        },
    },
    "components": {
        "schemas": {
            "Status": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "ok"},
                    "service": {"type": "string", "example": "student-performance-prediction"},
                },
            },
            "PredictRequest": {
                "type": "object",
                "required": FEATURES,
                "properties": {
                    "gender": {"type": "string", "enum": ["female", "male"], "example": "female"},
                    "race_ethnicity": {
                        "type": "string",
                        "enum": ["group A", "group B", "group C", "group D", "group E"],
                        "example": "group C",
                    },
                    "parental_level_of_education": {
                        "type": "string",
                        "enum": [
                            "some high school", "high school", "some college",
                            "associate's degree", "bachelor's degree", "master's degree",
                        ],
                        "example": "bachelor's degree",
                    },
                    "lunch": {"type": "string", "enum": ["standard", "free/reduced"], "example": "standard"},
                    "test_preparation_course": {
                        "type": "string", "enum": ["none", "completed"], "example": "completed"
                    },
                    "reading_score": {"type": "integer", "minimum": 0, "maximum": 100, "example": 90},
                    "writing_score": {"type": "integer", "minimum": 0, "maximum": 100, "example": 88},
                },
            },
            "PredictResponse": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string", "enum": ["PASS", "FAIL"], "example": "PASS"},
                    "prediction_code": {"type": "integer", "enum": [0, 1], "example": 1},
                },
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "example": "Missing required fields"},
                    "missing_fields": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Student Performance API - Swagger UI</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function () {
      window.ui = SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" });
    };
  </script>
</body>
</html>"""


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_performance_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/openapi.json")
def openapi_spec():
    return jsonify(OPENAPI_SPEC)


@app.get("/docs")
def swagger_ui():
    return SWAGGER_UI_HTML


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-performance-prediction"
    })


@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-performance-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{feature: data[feature] for feature in FEATURES}])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    prediction = "PASS" if prediction_code == 1 else "FAIL"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
