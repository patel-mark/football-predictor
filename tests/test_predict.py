import unittest
import pandas as pd
from src.predict import predict_fixtures

class TestPrediction(unittest.TestCase):
    def test_prediction_output(self):
        predictions = predict_fixtures('data/raw/fixtures.csv')
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertTrue('Home_Win_Prob' in predictions.columns)
        self.assertTrue('Predicted_Home_xG' in predictions.columns)
        self.assertEqual(len(predictions), 10, "Should predict 10 fixtures")