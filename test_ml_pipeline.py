import json
import os
import unittest

import joblib
import pandas as pd


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


class TestMLPipeline(unittest.TestCase):

    def test_source_dataset_exists(self):
        self.assertTrue(os.path.exists("StudentsPerformance.csv"))

    def test_source_dataset_has_1000_records(self):
        data = pd.read_csv("StudentsPerformance.csv")
        self.assertEqual(len(data), 1000)

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_performance_processed.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_performance_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_performance_model.pkl")
        prediction = model.predict(pd.DataFrame([HIGH_PERFORMER]))[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_performance_student(self):
        model = joblib.load("student_performance_model.pkl")
        prediction = model.predict(pd.DataFrame([HIGH_PERFORMER]))[0]
        self.assertEqual(int(prediction), 1)

    def test_low_performance_student(self):
        model = joblib.load("student_performance_model.pkl")
        prediction = model.predict(pd.DataFrame([LOW_PERFORMER]))[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
