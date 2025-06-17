import unittest
from src.train import train_model
from pathlib import Path

class TestTraining(unittest.TestCase):
    def test_model_creation(self):
        train_model()
        model_files = list(Path('models').glob('xgb_model_fold*.pkl'))
        self.assertGreaterEqual(len(model_files), 3, "At least 3 fold models should be created")
        self.assertTrue(Path('models/xgb_artifacts.pkl').exists(), "Artifacts file missing")