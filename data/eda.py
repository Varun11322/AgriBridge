import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.getcwd(), "data")
BATCHES_JSON_PATH = os.path.join(DATA_DIR, "synthetic_batches.json")

def load_and_engineer_features(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    df_events = pd.DataFrame(events)
    df_events['dt'] = pd.to_datetime(df_events['timestamp'])
    
    batches = []
    
    # Expected role transitions
    VALID_TRANSITIONS = {
        ("FARMER", "TRADER"),
        ("TRADER", "TRANSPORTER"),
        ("TRANSPORTER", "RETAILER")
    }
    
    for batch_id, group in df_events.groupby('batch_id'):
        group = group.sort_values('dt')
        
        is_anomalous = group['is_anomalous_chain'].iloc[0]
        anomaly_type = group['anomaly_type'].iloc[0]
        chain_len = len(group)
        distinct_actors = group['actor_id'].nunique()
        
        # Calculate time gaps and speeds
        time_gaps_hrs = []
        speeds_kmh = []
        role_valid_count = 0
        hash_valid_count = 0
        tx_hashes = list(group['tx_hash'])
        duplicate_tx_count = len(tx_hashes) - len(set(tx_hashes))
        
        prev_row = None
        for i, row in group.reset_index(drop=True).iterrows():
            if prev_row is not None:
                gap_sec = (row['dt'] - prev_row['dt']).total_seconds()
                gap_hrs = gap_sec / 3600.0
                time_gaps_hrs.append(gap_hrs)
                
                # Haversine distance
                lat1, lon1 = prev_row['lat'], prev_row['lon']
                lat2, lon2 = row['lat'], row['lon']
                
                # Simple Euclidean approx in km for lat/lon
                dist_km = np.sqrt(((lat2 - lat1)*111)**2 + ((lon2 - lon1)*111*np.cos(np.radians(lat1)))**2)
                
                if gap_hrs > 0:
                    speed = dist_km / gap_hrs
                else:
                    speed = 999.0 if dist_km > 0 else 0.0
                speeds_kmh.append(speed)
                
                # Role transition check
                if (prev_row['actor_role'], row['actor_role']) in VALID_TRANSITIONS:
                    role_valid_count += 1
                    
                # Hash link check
                if row['prev_hash'] != "0x" + "a" * 64: # check non-corrupted
                    hash_valid_count += 1
            else:
                hash_valid_count += 1
                
            prev_row = row
            
        avg_time_gap = float(np.mean(time_gaps_hrs)) if time_gaps_hrs else 0.0
        min_time_gap = float(np.min(time_gaps_hrs)) if time_gaps_hrs else 0.0
        max_speed = float(np.max(speeds_kmh)) if speeds_kmh else 0.0
        
        num_transitions = max(1, chain_len - 1)
        valid_role_ratio = role_valid_count / num_transitions
        hash_integrity_ratio = hash_valid_count / chain_len
        
        batches.append({
            "batch_id": batch_id,
            "chain_length": chain_len,
            "distinct_actors": distinct_actors,
            "avg_time_gap_hrs": avg_time_gap,
            "min_time_gap_hrs": min_time_gap,
            "max_speed_kmh": max_speed,
            "valid_role_ratio": valid_role_ratio,
            "hash_integrity_ratio": hash_integrity_ratio,
            "duplicate_tx_count": duplicate_tx_count,
            "is_anomalous": 1 if is_anomalous else 0,
            "anomaly_type": anomaly_type
        })
        
    return pd.DataFrame(batches)

if __name__ == "__main__":
    df_batches = load_and_engineer_features(BATCHES_JSON_PATH)
    
    print("================ EDA REPORT ================")
    print("\n--- 1. Class Balance ---")
    val_counts = df_batches['is_anomalous'].value_counts()
    val_pct = df_batches['is_anomalous'].value_counts(normalize=True) * 100
    print(f"Valid Chains (0):   {val_counts.get(0, 0)} ({val_pct.get(0, 0):.1f}%)")
    print(f"Broken Chains (1):  {val_counts.get(1, 0)} ({val_pct.get(1, 0):.1f}%)")
    
    print("\n--- Breakdown by Anomaly Type ---")
    print(df_batches[df_batches['is_anomalous'] == 1]['anomaly_type'].value_counts())
    
    print("\n--- 2. Feature Summary Statistics (Mean ± Std) ---")
    features = [
        "chain_length", "distinct_actors", "avg_time_gap_hrs", 
        "min_time_gap_hrs", "max_speed_kmh", "valid_role_ratio", 
        "hash_integrity_ratio", "duplicate_tx_count"
    ]
    
    summary = df_batches.groupby('is_anomalous')[features].agg(['mean', 'std']).T
    print(summary.round(3))
    
    print("\n--- 3. Correlation Matrix with Label (is_anomalous) ---")
    corrs = df_batches[features + ['is_anomalous']].corr()['is_anomalous'].sort_values(ascending=False)
    print(corrs.round(4))
