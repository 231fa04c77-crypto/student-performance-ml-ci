import unittest

from app import app

HIGH_PERFORMER = {
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 90,
    "writing_score": 88,
}

LOW_PERFORMER = {
    "gender": "male",
    "race_ethnicity": "group A",
    "parental_level_of_education": "high school",
    "lunch": "free/reduced",
    "test_preparation_course": "none",
    "reading_score": 35,
    "writing_score": 30,
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_high_performance_prediction(self):
        response = self.client.post("/predict", json=HIGH_PERFORMER)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")

    def test_low_performance_prediction(self):
        response = self.client.post("/predict", json=LOW_PERFORMER)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "FAIL")

    def test_openapi_spec(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        spec = response.get_json()
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertIn("/predict", spec["paths"])

    def test_swagger_ui_page(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"swagger-ui", response.data)

    def test_missing_field_validation(self):
        response = self.client.post("/predict", json={"gender": "female"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
