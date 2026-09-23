import os
import io
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_URL = "https://raw.githubusercontent.com/gramener/datasets/master/supply_chain.csv"
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw_supply_chain.csv")
BATCHES_CSV_PATH = os.path.join(DATA_DIR, "synthetic_batches.csv")
BATCHES_JSON_PATH = os.path.join(DATA_DIR, "synthetic_batches.json")

print(f"Downloading real supply chain dataset from {RAW_DATA_URL}...")
r = requests.get(RAW_DATA_URL, timeout=30)
r.raise_for_status()

with open(RAW_CSV_PATH, "w", encoding="utf-8") as f:
    f.write(r.text)

print(f"Saved raw dataset to {RAW_CSV_PATH}")

df_raw = pd.read_csv(RAW_CSV_PATH)
print("Raw Dataset shape:", df_raw.shape)

# Grounded Indian Agri Locations & Actor Profiles
PRODUCE_TYPES = ["Basmati Rice", "Nashik Tomatoes", "Nagpur Oranges", "Alphonso Mangoes", "Kashmir Apples", "Punjab Wheat"]

LOCATIONS = [
    {"name": "Nashik Farm", "lat": 20.0059, "lon": 73.7898},
    {"name": "Vashi APMC Mandi", "lat": 19.0770, "lon": 73.0078},
    {"name": "Bhiwandi Cold Storage Hub", "lat": 19.2968, "lon": 73.0629},
    {"name": "Mumbai Central Retail Market", "lat": 19.0760, "lon": 72.8777},
    {"name": "Pune Agri Warehouse", "lat": 18.5204, "lon": 73.8567},
    {"name": "Nagpur Mandi Hub", "lat": 21.1458, "lon": 79.0882}
]

def process_real_data_to_event_chains(df, num_batches=1000):
    np.random.seed(42)
    all_events = []
    
    # Expand raw dataset samples to reach target 1000 batches
    raw_records = df.to_dict(orient="records")
    
    for i in range(num_batches):
        raw_item = raw_records[i % len(raw_records)]
        batch_id = f"AG-{2000 + i}"
        produce = PRODUCE_TYPES[i % len(PRODUCE_TYPES)]
        
        # ~15% overall anomaly rate ground truth (1 in every 6-7 chains)
        is_broken = bool(i % 6.5 < 1)
        anomaly_type = None
        if is_broken:
            anomaly_types = ["skipped_role", "backward_timestamp", "speed_violation", "broken_hash", "duplicate_tx"]
            anomaly_type = anomaly_types[i % len(anomaly_types)]
        
        # Chain setup
        start_time = datetime.now(timezone.utc) - timedelta(days=np.random.randint(2, 30))
        curr_time = start_time
        prev_hash = "0x" + hashlib.sha256(f"GENESIS_{batch_id}".encode()).hexdigest()
        
        # Roles sequence
        roles = ["FARMER", "TRADER", "TRANSPORTER", "RETAILER"]
        if anomaly_type == "skipped_role":
            # Direct Farmer to Retailer (skip trader/transporter)
            roles = ["FARMER", "RETAILER"]
        
        # Unique actors
        farmer_id = f"0x{hashlib.md5(f'farmer_{i}'.encode()).hexdigest()[:8]}"
        supplier_val = raw_item.get('Supplier name', i)
        carrier_val = raw_item.get('Shipping carriers', i)
        trader_id = f"0x{hashlib.md5(f'trader_{supplier_val}'.encode()).hexdigest()[:8]}"
        transporter_id = f"0x{hashlib.md5(f'transporter_{carrier_val}'.encode()).hexdigest()[:8]}"
        retailer_id = f"0x{hashlib.md5(f'retailer_{i}'.encode()).hexdigest()[:8]}"
        
        actor_map = {
            "FARMER": (farmer_id, "HARVEST", LOCATIONS[0]),
            "TRADER": (trader_id, "TRANSFER", LOCATIONS[1]),
            "TRANSPORTER": (transporter_id, "TRANSPORT", LOCATIONS[2]),
            "RETAILER": (retailer_id, "RETAIL_RECEIVE", LOCATIONS[3])
        }
        
        chain_tx_hashes = []
        
        for idx, role in enumerate(roles):
            actor_id, event_type, loc = actor_map[role]
            
            # Generate tx_hash
            tx_hash = "0x" + hashlib.sha256(f"{batch_id}_{idx}_{curr_time.isoformat()}".encode()).hexdigest()[:32]
            
            if anomaly_type == "duplicate_tx" and idx == len(roles) - 1:
                tx_hash = chain_tx_hashes[0] # Reuse first tx_hash
            chain_tx_hashes.append(tx_hash)
            
            # Check for broken_hash anomaly
            event_prev_hash = prev_hash
            if anomaly_type == "broken_hash" and idx == len(roles) - 1:
                event_prev_hash = "0x" + "a" * 64 # Corrupted hash link
                
            event_obj = {
                "batch_id": batch_id,
                "produce": produce,
                "event_type": event_type,
                "actor_id": actor_id,
                "actor_role": role,
                "timestamp": curr_time.isoformat(),
                "prev_hash": event_prev_hash,
                "lat": loc["lat"],
                "lon": loc["lon"],
                "location": loc["name"],
                "tx_hash": tx_hash,
                "real_supplier": str(raw_item.get("Supplier name", "AgriSupplier")),
                "real_carrier": str(raw_item.get("Shipping carriers", "CarrierA")),
                "real_inspection": str(raw_item.get("Inspection results", "Pass")),
                "real_defect_rate": float(raw_item.get("Defect rates", 0.0)),
                "is_anomalous_chain": is_broken,
                "anomaly_type": anomaly_type if is_broken else "none"
            }
            all_events.append(event_obj)
            
            # Calculate next prev_hash
            hash_payload = json.dumps(event_obj, sort_keys=True)
            prev_hash = "0x" + hashlib.sha256(hash_payload.encode()).hexdigest()
            
            # Time progression based on real shipping time & transit speed
            if anomaly_type == "backward_timestamp" and idx == len(roles) - 1:
                curr_time -= timedelta(minutes=np.random.randint(20, 120)) # Subtle backward timestamp (-20m to -2h)
            elif anomaly_type == "speed_violation" and idx == 2:
                curr_time += timedelta(minutes=15) # 200 km covered in 15 mins (implied speed > 500 km/h)
            else:
                # Realistic gap between stages: 4-18 hours
                curr_time += timedelta(hours=float(np.random.uniform(4, 18)))
                
    return all_events

events = process_real_data_to_event_chains(df_raw, num_batches=1000)
df_events = pd.DataFrame(events)

print(f"Generated {len(df_events)} total events across 1000 batches.")
df_events.to_csv(BATCHES_CSV_PATH, index=False)

with open(BATCHES_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(events, f, indent=2)

print(f"Saved dataset files:\n  CSV: {BATCHES_CSV_PATH}\n  JSON: {BATCHES_JSON_PATH}")
