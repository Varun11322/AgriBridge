# Walkthrough — Traceability Agent (AgriBridge)

Completed the end-to-end implementation of the **Traceability Agent** for AgriBridge. The agent scores the integrity of produce batch custody chains (FARMER → TRADER → TRANSPORTER → RETAILER) using rule-based structural verification combined with a `scikit-learn` Isolation Forest machine learning model.

---

## Key Achievements

### 1. Real Dataset Integration & Batch Event Chain Generator
- **Real Public Source**: Downloaded `raw_supply_chain.csv` from Gramener GitHub repository.
- **Processed Event Chains**: Generated **1,000 produce batch chains** (3,936 total events) saved to:
  - `data/synthetic_batches.csv`
  - `data/synthetic_batches.json`
- **Class Balance**: 846 Valid Chains (84.6%), 154 Broken Chains (15.4%).

### 2. Standalone Rule-Based Verifier (`src/agents/traceability/rules.py`)
- **Role Sequence Check**: Validates transition flow `FARMER` → `TRADER` → `TRANSPORTER` → `RETAILER`.
- **Cryptographic Hash Link**: Verifies SHA-256 `prev_hash` across linked events.
- **Timestamp Order**: Detects backward timestamps (backdating tampering).
- **Tx Hash Uniqueness**: Flags duplicate `tx_hash` instances.
- **Speed Plausibility**: Detects impossible truck transport speeds (> 110 km/h).

### 3. ML Feature Extraction & Isolation Forest (`src/agents/traceability/features.py` & `model.py`)
- Engineered 9 batch-level features (`avg_time_gap_hrs`, `min_time_gap_hrs`, `max_speed_kmh`, `valid_role_ratio`, `hash_integrity_ratio`, `duplicate_tx_count`, `actor_reputation_avg`, `chain_length`, `distinct_actors`).
- Trained `sklearn.ensemble.IsolationForest` and saved trained model artifact to `src/agents/traceability/models/isolation_forest.joblib`.
- Integrated combined confidence formula:
  $$\text{confidence\_score} = \text{clip}\left(0.6 \times \text{rule\_score} + 0.4 \times \text{ml\_anomaly\_score}, 0, 100\right)$$

### 4. FastAPI REST Endpoint (`src/agents/traceability/api.py`)
- Endpoint: `POST /verify-batch/{batch_id}`
- Output Contract:
```json
{
  "batch_id": "AG-2000",
  "status": "FLAGGED",
  "confidence_score": 45,
  "chain_length": 2,
  "issues": [
    "Invalid role sequence transition: FARMER -> RETAILER"
  ]
}
```

---

## Evaluation Results (`src/agents/traceability/evaluate.py`)

Run evaluation: `python -m src.agents.traceability.evaluate`

```
================ EVALUATION METRICS ================

--- Classification Report (0 = Valid, 1 = Broken Chain) ---
              precision    recall  f1-score   support

 Valid Chain       0.97      1.00      0.98       846
Broken Chain       1.00      0.81      0.89       154

    accuracy                           0.97      1000
   macro avg       0.98      0.90      0.94      1000
weighted avg       0.97      0.97      0.97      1000

--- Confusion Matrix ---
True Valid (TN):  846   | False Flagged (FP): 0
False Valid (FN): 30    | True Flagged (TP):  124

Overall Performance:
  Precision: 100.00%
  Recall:    80.52%
  F1 Score:  89.21%

--- Confidence Score Distribution ---
Valid Batches Confidence Mean ± Std:  87.5% ± 3.0%
Broken Batches Confidence Mean ± Std: 53.0% ± 16.5%
====================================================
```

---

## How to Run

1. **Run Model Evaluation & Retrain**:
   ```bash
   python -m src.agents.traceability.evaluate
   ```

2. **Start FastAPI Service**:
   ```bash
   python -m uvicorn src.agents.traceability.api:app --host 0.0.0.0 --port 8000
   ```

3. **Verify a Batch via HTTP**:
   ```bash
   curl -X POST http://localhost:8000/verify-batch/AG-2006
   ```
