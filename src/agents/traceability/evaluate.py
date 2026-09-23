import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from .features import extract_batch_features, FEATURE_COLUMNS
from .model import TraceabilityModel, get_traceability_model

DATA_DIR = os.path.join(os.getcwd(), "data")
BATCHES_JSON_PATH = os.path.join(DATA_DIR, "synthetic_batches.json")

def evaluate_pipeline():
    print("Loading synthetic dataset for model training and evaluation...")
    if not os.path.exists(BATCHES_JSON_PATH):
        raise FileNotFoundError(f"Dataset not found at {BATCHES_JSON_PATH}. Run fetch_and_process_real_dataset.py first.")
        
    with open(BATCHES_JSON_PATH, "r", encoding="utf-8") as f:
        events = json.load(f)
        
    df_events = pd.DataFrame(events)
    
    batch_records = []
    for batch_id, group in df_events.groupby('batch_id'):
        group_events = group.to_dict(orient="records")
        is_anomalous = group['is_anomalous_chain'].iloc[0]
        feats = extract_batch_features(group_events)
        feats['batch_id'] = batch_id
        feats['is_anomalous'] = 1 if is_anomalous else 0
        feats['events'] = group_events
        batch_records.append(feats)
        
    df_batches = pd.DataFrame(batch_records)
    print(f"Loaded {len(df_batches)} batch custody chains.")
    
    # Train IsolationForest model on dataset
    model_obj = TraceabilityModel()
    model_obj.train(df_batches)
    
    y_true = df_batches['is_anomalous'].values # 1 = broken, 0 = valid
    y_pred = []
    confidence_scores = []
    
    for _, row in df_batches.iterrows():
        res = model_obj.evaluate_batch(row['batch_id'], row['events'])
        pred_label = 1 if res['status'] == 'FLAGGED' else 0
        y_pred.append(pred_label)
        confidence_scores.append((row['is_anomalous'], res['confidence_score']))
        
    y_pred = np.array(y_pred)
    
    print("\n================ EVALUATION METRICS ================")
    print("\n--- Classification Report (0 = Valid, 1 = Broken Chain) ---")
    print(classification_report(y_true, y_pred, target_names=["Valid Chain", "Broken Chain"]))
    
    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(f"True Valid (TN):  {cm[0][0]:<5} | False Flagged (FP): {cm[0][1]}")
    print(f"False Valid (FN): {cm[1][0]:<5} | True Flagged (TP):  {cm[1][1]}")
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"\nOverall Performance:")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    df_conf = pd.DataFrame(confidence_scores, columns=['is_anomalous', 'confidence'])
    print("\n--- Confidence Score Distribution ---")
    print("Valid Batches Confidence Mean ± Std: ", f"{df_conf[df_conf['is_anomalous']==0]['confidence'].mean():.1f}% ± {df_conf[df_conf['is_anomalous']==0]['confidence'].std():.1f}%")
    print("Broken Batches Confidence Mean ± Std:", f"{df_conf[df_conf['is_anomalous']==1]['confidence'].mean():.1f}% ± {df_conf[df_conf['is_anomalous']==1]['confidence'].std():.1f}%")
    print("====================================================")

if __name__ == "__main__":
    evaluate_pipeline()
