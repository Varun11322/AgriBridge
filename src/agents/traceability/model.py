import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .rules import evaluate_structural_rules
from .features import extract_batch_features, FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")

class TraceabilityModel:
    def __init__(self):
        self.model = None
        self._load_or_init_model()
        
    def _load_or_init_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                return
            except Exception as e:
                print(f"Failed to load model from {MODEL_PATH}: {e}")
                
        # Initialize default IsolationForest model
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.15, # ~15% anomalies
            random_state=42
        )
        
    def train(self, df_features):
        """
        Trains IsolationForest model on pandas DataFrame of batch features.
        """
        X = df_features[FEATURE_COLUMNS]
        self.model.fit(X)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def predict_batch_anomaly_score(self, feature_dict):
        """
        Computes normalized ML anomaly score (0-100%, where 100% means fully normal/high confidence).
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return 85.0 # fallback if un-fitted
            
        X = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
        
        # decision_function gives negative values for anomalies, positive for normal
        raw_score = self.model.decision_function(X)[0]
        
        # Scale decision_function score (-0.3 to +0.3) into 0-100% confidence scale
        # Score >= +0.1 maps to ~95-100%, Score <= -0.15 maps to <40%
        normalized_score = 50.0 + (raw_score * 250.0)
        return float(max(0.0, min(100.0, normalized_score)))

    def evaluate_batch(self, batch_id, events):
        """
        Evaluates a produce batch history through rule-based & ML anomaly detection pipeline.
        
        Returns exact spec JSON output contract.
        """
        # 1. Rule-based structural evaluation
        rule_eval = evaluate_structural_rules(events)
        rule_score = rule_eval["rule_score"]
        issues = rule_eval["issues"]
        failed_rules = rule_eval["failed_rules"]
        
        # 2. ML Anomaly Layer
        features = extract_batch_features(events)
        ml_anomaly_score = self.predict_batch_anomaly_score(features)
        
        # 3. Weighted Confidence Formula from Spec:
        # confidence = 0.6 * rule_score + 0.4 * ml_anomaly_score
        confidence_score = (0.6 * rule_score) + (0.4 * ml_anomaly_score)
        
        # Severe rule violations penalize score further
        if failed_rules:
            confidence_score = min(confidence_score, 45.0)
            
        confidence_score = int(round(max(0.0, min(100.0, confidence_score))))
        
        # 4. Status Determination
        chain_length = len(events)
        roles = [e.get("actor_role") for e in events]
        
        if failed_rules or confidence_score < 65:
            status = "FLAGGED"
        elif chain_length < 4 or "RETAILER" not in roles:
            status = "INCOMPLETE"
        else:
            status = "VERIFIED"
            
        return {
            "batch_id": str(batch_id),
            "status": status,
            "confidence_score": confidence_score,
            "chain_length": chain_length,
            "issues": issues
        }

# Global singleton model instance
_instance = None

def get_traceability_model():
    global _instance
    if _instance is None:
        _instance = TraceabilityModel()
    return _instance
