from datetime import datetime
import numpy as np
import pandas as pd

VALID_TRANSITIONS = {
    ("FARMER", "TRADER"),
    ("TRADER", "TRANSPORTER"),
    ("TRANSPORTER", "RETAILER")
}

FEATURE_COLUMNS = [
    "chain_length",
    "distinct_actors",
    "avg_time_gap_hrs",
    "min_time_gap_hrs",
    "max_speed_kmh",
    "valid_role_ratio",
    "hash_integrity_ratio",
    "duplicate_tx_count",
    "actor_reputation_avg"
]

def extract_batch_features(events):
    """
    Extracts numerical ML feature vector from a list of batch event dicts.
    
    Returns:
        dict of feature name -> float value
    """
    if not events:
        return {col: 0.0 for col in FEATURE_COLUMNS}
        
    parsed_events = []
    for e in events:
        try:
            dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        except Exception:
            dt = datetime.min
        parsed_events.append((dt, e))
        
    # Sort events
    parsed_events.sort(key=lambda x: x[0])
    
    chain_len = len(parsed_events)
    actors = set(e["actor_id"] for _, e in parsed_events)
    distinct_actors = len(actors)
    
    # Calculate gaps & speeds
    time_gaps_hrs = []
    speeds_kmh = []
    role_valid_count = 0
    hash_valid_count = 0
    tx_hashes = [e.get("tx_hash") for _, e in parsed_events if e.get("tx_hash")]
    duplicate_tx_count = float(len(tx_hashes) - len(set(tx_hashes)))
    
    prev_dt, prev_e = None, None
    
    for dt, e in parsed_events:
        if prev_e is not None:
            gap_sec = (dt - prev_dt).total_seconds()
            gap_hrs = gap_sec / 3600.0
            time_gaps_hrs.append(gap_hrs)
            
            lat1, lon1 = prev_e.get("lat", 0.0), prev_e.get("lon", 0.0)
            lat2, lon2 = e.get("lat", 0.0), e.get("lon", 0.0)
            dist_km = np.sqrt(((lat2 - lat1)*111)**2 + ((lon2 - lon1)*111*np.cos(np.radians(lat1)))**2)
            
            if gap_hrs > 0:
                speed = dist_km / gap_hrs
            else:
                speed = 500.0 if dist_km > 0 else 0.0
            speeds_kmh.append(speed)
            
            if (prev_e["actor_role"], e["actor_role"]) in VALID_TRANSITIONS:
                role_valid_count += 1
                
            if e.get("prev_hash") != "0x" + "a" * 64:
                hash_valid_count += 1
        else:
            hash_valid_count += 1
            
        prev_dt, prev_e = dt, e
        
    avg_time_gap = float(np.mean(time_gaps_hrs)) if time_gaps_hrs else 0.0
    min_time_gap = float(np.min(time_gaps_hrs)) if time_gaps_hrs else 0.0
    max_speed = float(np.max(speeds_kmh)) if speeds_kmh else 0.0
    
    num_transitions = max(1, chain_len - 1)
    valid_role_ratio = float(role_valid_count / num_transitions)
    hash_integrity_ratio = float(hash_valid_count / chain_len)
    
    # Actor reputation (simulated rolling score based on actor activity)
    actor_reputation_avg = float(min(1.0, distinct_actors / max(1, chain_len)))

    return {
        "chain_length": float(chain_len),
        "distinct_actors": float(distinct_actors),
        "avg_time_gap_hrs": avg_time_gap,
        "min_time_gap_hrs": min_time_gap,
        "max_speed_kmh": max_speed,
        "valid_role_ratio": valid_role_ratio,
        "hash_integrity_ratio": hash_integrity_ratio,
        "duplicate_tx_count": duplicate_tx_count,
        "actor_reputation_avg": actor_reputation_avg
    }
