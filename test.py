"""
Tests for CI/CD pipeline
"""

import unittest
import pandas as pd
import os
import joblib
import json

class TestPipeline(unittest.TestCase):
    
    def test_1_data_exists(self):
        """Test data file exists"""
        self.assertTrue(os.path.exists('data/UCI_Credit_Card.csv'))
    
    def test_2_data_loads(self):
        """Test data loads correctly"""
        df = pd.read_csv('data/UCI_Credit_Card.csv')
        self.assertGreater(len(df), 10000)
        self.assertGreater(len(df.columns), 10)
    
    def test_3_models_exist(self):
        """Test models were created"""
        self.assertTrue(os.path.exists('models/logistic_model.pkl'))
        self.assertTrue(os.path.exists('models/scaler.pkl'))
        self.assertTrue(os.path.exists('models/metrics.json'))
    
    def test_4_model_loads(self):
        """Test model loads"""
        model = joblib.load('models/logistic_model.pkl')
        self.assertIsNotNone(model)
    
    def test_5_metrics_valid(self):
        """Test metrics are reasonable"""
        with open('models/metrics.json') as f:
            metrics = json.load(f)
        
        # Check structure
        self.assertIn('train', metrics)
        self.assertIn('validation', metrics)
        self.assertIn('test', metrics)
        
        # Check test metrics
        test = metrics['test']
        self.assertGreater(test['accuracy'], 0.7)
        self.assertGreater(test['roc_auc'], 0.6)
        self.assertLessEqual(test['accuracy'], 1.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)


