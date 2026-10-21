
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

# --------------------------------------------------
# 1. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

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

logger.info("event=application_starting")


# --------------------------------------------------
# 2. REQUEST LOGGING MIDDLEWARE
# --------------------------------------------------

@app.before_request
def log_request_start():
    g.request_id = str(uuid.uuid4())
    g.request_start_time = time.time()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "request_start_time", time.time())
    duration_ms = round((time.time() - start_time) * 1000, 2)

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "event=request_completed request_id=%s "
        "method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


# --------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------

def load_model():
    logger.info("event=model_load_started")

    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed reason=model_file_missing"
        )
        raise FileNotFoundError(
            "student_performance_model.pkl was not found. "
            "Run the training pipeline first."
        )

    model = joblib.load(MODEL_PATH)

    logger.info("event=model_loaded")
    return model


# --------------------------------------------------
# 4. OPENAPI DOCUMENTATION
# --------------------------------------------------

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
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Status"
                                }
                            }
                        },
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
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Status"
                                }
                            }
                        },
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
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/PredictRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Prediction result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictResponse"
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "Missing or invalid request body",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        },
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
                    "status": {
                        "type": "string",
                        "example": "ok",
                    },
                    "service": {
                        "type": "string",
                        "example": "student-performance-prediction",
                    },
                },
            },
            "PredictRequest": {
                "type": "object",
                "required": FEATURES,
                "properties": {
                    "gender": {
                        "type": "string",
                        "enum": ["female", "male"],
                        "example": "female",
                    },
                    "race_ethnicity": {
                        "type": "string",
                        "enum": [
                            "group A",
                            "group B",
                            "group C",
                            "group D",
                            "group E",
                        ],
                        "example": "group C",
                    },
                    "parental_level_of_education": {
                        "type": "string",
                        "enum": [
                            "some high school",
                            "high school",
                            "some college",
                            "associate's degree",
                            "bachelor's degree",
                            "master's degree",
                        ],
                        "example": "bachelor's degree",
                    },
                    "lunch": {
                        "type": "string",
                        "enum": ["standard", "free/reduced"],
                        "example": "standard",
                    },
                    "test_preparation_course": {
                        "type": "string",
                        "enum": ["none", "completed"],
                        "example": "completed",
                    },
                    "reading_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "example": 90,
                    },
                    "writing_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "example": 88,
                    },
                },
            },
            "PredictResponse": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "string",
                        "enum": ["PASS", "FAIL"],
                        "example": "PASS",
                    },
                    "prediction_code": {
                        "type": "integer",
                        "enum": [0, 1],
                        "example": 1,
                    },
                },
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "example": "Missing required fields",
                    },
                    "missing_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
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
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function () {
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui"
      });
    };
  </script>
</body>
</html>"""


# --------------------------------------------------
# 5. API ROUTES
# --------------------------------------------------

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
        "service": "student-performance-prediction",
    })


@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-performance-prediction",
    })


# --------------------------------------------------
# 6. PREDICTION WITH ERROR TRACKING
# --------------------------------------------------

@app.post("/predict")
def predict():
    request_id = getattr(g, "request_id", "unknown")

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or not data:
            logger.warning(
                "event=prediction_validation_failed "
                "request_id=%s reason=invalid_or_missing_json",
                request_id,
            )
            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing_fields = [
            feature for feature in FEATURES if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "event=prediction_validation_failed "
                "request_id=%s reason=missing_fields fields=%s",
                request_id,
                ",".join(missing_fields),
            )
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }), 400

        # Validate scores without logging the submitted values.
        for field in ["reading_score", "writing_score"]:
            value = data[field]

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 100
            ):
                logger.warning(
                    "event=prediction_validation_failed "
                    "request_id=%s reason=invalid_score field=%s",
                    request_id,
                    field,
                )
                return jsonify({
                    "error": (
                        f"{field} must be an integer between 0 and 100"
                    )
                }), 400

        sample = pd.DataFrame([
            {feature: data[feature] for feature in FEATURES}
        ])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_succeeded request_id=%s result=%s",
            request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )
        return jsonify({
            "error": "Prediction failed due to an internal server error",
            "request_id": request_id,
        }), 500


# --------------------------------------------------
# 7. UNEXPECTED SERVER ERROR TRACKING
# --------------------------------------------------

@app.errorhandler(500)
def handle_internal_server_error(error):
    request_id = getattr(g, "request_id", "unknown")

    logger.error(
        "event=internal_server_error request_id=%s",
        request_id,
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": request_id,
    }), 500


# --------------------------------------------------
# 8. APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting_server port=5000")
    app.run(host="0.0.0.0", port=5000)
