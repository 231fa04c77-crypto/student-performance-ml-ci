# Student Performance ML CI/CD

Predicts whether a student PASSES (math score >= 50) using `StudentsPerformance.csv`.

Features: gender, race_ethnicity, parental_level_of_education, lunch,
test_preparation_course, reading_score, writing_score.

Pipeline: train -> quality gate (accuracy >= 0.85) -> ML tests -> API tests
-> release candidate artifact -> Docker build + container tests.
