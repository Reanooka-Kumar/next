import os
import numpy as np
import joblib
import json

try:
    from data_preprocessing import preprocess_single_student
except ImportError:
    from src.data_preprocessing import preprocess_single_student

class PlacementPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        self.load_assets()
        
    def load_assets(self):
        """Loads trained model assets and metadata."""
        classifier_path = os.path.join(self.models_dir, "best_classifier.joblib")
        regressor_path = os.path.join(self.models_dir, "salary_regressor.joblib")
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        features_path = os.path.join(self.models_dir, "feature_names.joblib")
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        
        if os.path.exists(classifier_path):
            self.classifier = joblib.load(classifier_path)
        if os.path.exists(regressor_path):
            self.regressor = joblib.load(regressor_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
                
    def is_ready(self):
        """Checks if all required model assets are loaded."""
        return (self.classifier is not None and 
                self.regressor is not None and 
                self.scaler is not None and 
                self.feature_names is not None)

    def predict_student(self, student_dict):
        """
        Takes student profile dictionary and returns placement predictions and expected salary.
        """
        if not self.is_ready():
            # Reload to see if they were generated in background
            self.load_assets()
            if not self.is_ready():
                return {
                    'error': 'Model assets not found. Please train models first.'
                }
                
        # Preprocess student profile
        X_scaled, mapped_features, feature_names = preprocess_single_student(student_dict, self.scaler)
        
        # 1. Classification (Placement Status)
        pred_class = self.classifier.predict(X_scaled)[0]
        probs = self.classifier.predict_proba(X_scaled)[0]
        placed_prob = float(probs[1]) # Probability of Placed (class 1)
        
        status = "Placed" if pred_class == 1 else "Not Placed"
        
        # Confidence Score: max probability of the predicted class (ranges from 50% to 100%)
        confidence_score = float(max(placed_prob, 1.0 - placed_prob) * 100)
        
        # 2. Regression (Expected Salary)
        expected_salary = float(self.regressor.predict(X_scaled)[0])
        
        # If student is predicted 'Not Placed', expected salary can be shown as a potential package
        # but let's represent it as potential expected salary.
        
        # Salary Range (95% Confidence Interval)
        residual_std = self.metrics.get('salary_metrics', {}).get('residual_std', 1.2)
        margin_of_error = 1.96 * residual_std
        
        salary_min = max(2.0, expected_salary - margin_of_error)
        salary_max = expected_salary + margin_of_error
        
        return {
            'placement_status': status,
            'placement_probability': placed_prob * 100, # convert to percentage
            'confidence_score': confidence_score,
            'expected_salary': expected_salary,
            'salary_range_min': salary_min,
            'salary_range_max': salary_max,
            'confidence_interval_width': margin_of_error * 2,
            'mapped_features': mapped_features,
            'scaled_features_array': X_scaled,
            'feature_names': feature_names
        }
