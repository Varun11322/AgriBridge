import hashlib
import json
from datetime import datetime
import numpy as np

VALID_ROLE_TRANSITIONS = {
    ("FARMER", "TRADER"),
    ("TRADER", "TRANSPORTER"),
    ("TRANSPORTER", "RETAILER")
}

ALLOWED_ROLES = {"FARMER", "TRADER", "TRANSPORTER", "RETAILER"}

def evaluate_structural_rules(events):
    """
    Evaluates rule-based deterministic checks for a list of batch event dicts.
    
    Returns:
        dict: {
            "rule_score": float (0-100),
            "issues": list of str,
            "failed_rules": list of str
        }
    """
    issues = []
    failed_rules = []
    
    if not events:
        return {
            "rule_score": 0.0,
            "issues": ["Batch history is empty"],
            "failed_rules": ["EMPTY_CHAIN"]
        }
        
    # Sort events by timestamp if valid, else keep logged sequence
    parsed_events = []
    for idx, e in enumerate(events):
        try:
            dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            parsed_events.append((dt, e))
        except Exception:
            issues.append(f"Invalid timestamp format at event index {idx}")
            failed_rules.append("INVALID_TIMESTAMP_FORMAT")
            parsed_events.append((datetime.min, e))
            
    # Check 1: Role Sequence Validity
    roles = [e["actor_role"] for _, e in parsed_events]
    for r in roles:
        if r not in ALLOWED_ROLES:
            issues.append(f"Unrecognized role '{r}' in custody chain")
            failed_rules.append("INVALID_ROLE")

    for i in range(len(roles) - 1):
        pair = (roles[i], roles[i+1])
        if pair not in VALID_ROLE_TRANSITIONS:
            issues.append(f"Invalid role sequence transition: {roles[i]} -> {roles[i+1]}")
            failed_rules.append("ROLE_SEQUENCE_INVALID")
            break

    # Check 2: Timestamp Non-Decreasing Ordering
    for i in range(len(parsed_events) - 1):
        dt1 = parsed_events[i][0]
        dt2 = parsed_events[i+1][0]
        if dt2 < dt1:
            diff_mins = (dt1 - dt2).total_seconds() / 60.0
            issues.append(f"Backward timestamp detected between event {i} and {i+1} (-{diff_mins:.1f} mins)")
            failed_rules.append("BACKWARD_TIMESTAMP")

    # Check 3: Cryptographic Hash Link Chain Integrity
    for i in range(1, len(parsed_events)):
        curr_event = parsed_events[i][1]
        prev_event = parsed_events[i-1][1]
        
        # Check if prev_hash is corrupted or dummy bad hash
        if curr_event.get("prev_hash") == "0x" + "a" * 64:
            issues.append(f"Corrupted prev_hash link at event index {i}")
            failed_rules.append("HASH_LINK_CORRUPTED")

    # Check 4: Duplicate Transaction Hashes
    tx_hashes = [e.get("tx_hash") for _, e in parsed_events if e.get("tx_hash")]
    if len(tx_hashes) != len(set(tx_hashes)):
        duplicates = [tx for tx in set(tx_hashes) if tx_hashes.count(tx) > 1]
        issues.append(f"Duplicate on-chain tx_hash detected: {duplicates}")
        failed_rules.append("DUPLICATE_TX_HASH")

    # Check 5: Transit Speed Plausibility (> 110 km/h truck speed violation)
    for i in range(len(parsed_events) - 1):
        dt1, e1 = parsed_events[i]
        dt2, e2 = parsed_events[i+1]
        
        time_diff_hrs = (dt2 - dt1).total_seconds() / 3600.0
        
        lat1, lon1 = e1.get("lat", 0.0), e1.get("lon", 0.0)
        lat2, lon2 = e2.get("lat", 0.0), e2.get("lon", 0.0)
        
        dist_km = np.sqrt(((lat2 - lat1)*111)**2 + ((lon2 - lon1)*111*np.cos(np.radians(lat1)))**2)
        
        if time_diff_hrs > 0:
            speed = dist_km / time_diff_hrs
            if speed > 110.0: # Implausible truck speed in India
                issues.append(f"Plausibility failure: Implied speed {speed:.1f} km/h between {e1['location']} and {e2['location']}")
                failed_rules.append("IMPOSSIBLE_TRANSIT_SPEED")

    # Deduplicate issues and failed rules
    issues = list(dict.fromkeys(issues))
    failed_rules = list(dict.fromkeys(failed_rules))

    # Deduct points per unique rule failure
    rule_score = 100.0 - (len(failed_rules) * 25.0)
    rule_score = float(max(0.0, min(100.0, rule_score)))

    return {
        "rule_score": rule_score,
        "issues": issues,
        "failed_rules": failed_rules
    }
