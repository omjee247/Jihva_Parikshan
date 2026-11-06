"""
Feedback Sanitizer, Adaptive Calibration Store & Patient Review Engine.
Backend-only sanitization & KNN calibration (transparent to end-users).
"""

import os
import re
import json
import sqlite3
import datetime
import numpy as np
from typing import Dict, Any, List, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "jihva_feedback.db")

AYURVEDIC_DOMAIN_KEYWORDS = {
    "tongue", "jihva", "vata", "pitta", "kapha", "ama", "agni", "coating", "lepa",
    "fissure", "crack", "groove", "red", "pale", "pink", "white", "yellow", "brown",
    "tooth", "indentation", "swollen", "thin", "dry", "moist", "saliva", "digestion",
    "acidity", "constipation", "mucus", "ulcer", "burning", "vaidya", "ayurveda",
    "dosha", "prakriti", "vikriti", "tip", "root", "border", "spleen", "stomach"
}

BANNED_SPAM_PATTERNS = [
    r"https?://", r"www\.", r"buy now", r"crypto", r"casino", r"loan", r"discount",
    r"(.)\1{5,}",
]


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            patient_age INTEGER,
            patient_gender TEXT,
            features_json TEXT NOT NULL,
            predicted_dosha TEXT NOT NULL,
            corrected_dosha TEXT NOT NULL,
            corrected_colour TEXT NOT NULL,
            corrected_coating TEXT NOT NULL,
            corrected_fissures TEXT NOT NULL,
            raw_notes TEXT,
            cleaned_notes TEXT,
            relevance_score REAL NOT NULL,
            is_accepted INTEGER NOT NULL,
            rejection_reason TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            rating INTEGER NOT NULL,
            dominant_dosha TEXT,
            review_text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def sanitize_and_validate_feedback(
    features: List[float],
    corrected_dosha: str,
    corrected_colour: str,
    corrected_coating: str,
    corrected_fissures: str,
    raw_notes: str
) -> Tuple[bool, float, str, str]:
    notes = (raw_notes or "").strip()

    for pat in BANNED_SPAM_PATTERNS:
        if re.search(pat, notes, flags=re.IGNORECASE):
            return False, 0.0, "", "Rejected: Promotional or invalid text."

    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", notes)
    cleaned = re.sub(r"\b\d{10,12}\b", "[REDACTED_PHONE]", cleaned)

    sentences = re.split(r"[.!?\n]+", cleaned)
    relevant_sentences = []
    matched_keywords = 0
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        words = set(re.findall(r"[a-zA-Z]+", s_clean.lower()))
        overlap = words.intersection(AYURVEDIC_DOMAIN_KEYWORDS)
        if overlap:
            matched_keywords += len(overlap)
            relevant_sentences.append(s_clean)

    cleaned_clinical_notes = ". ".join(relevant_sentences)

    mean_A = features[1] if len(features) > 1 else 145.0
    coating_pct = features[4] if len(features) > 4 else 15.0

    if "Absent" in corrected_coating and coating_pct > 55.0:
        return False, 0.15, cleaned_clinical_notes, "Filtered out due to coating mismatch."
    if "Thick" in corrected_coating and coating_pct < 3.0:
        return False, 0.20, cleaned_clinical_notes, "Filtered out due to coating mismatch."
    if corrected_colour.startswith("Red") and mean_A < 134.0:
        return False, 0.20, cleaned_clinical_notes, "Filtered out due to colour mismatch."

    relevance_score = min(1.0, round(0.65 + 0.07 * min(matched_keywords, 5), 2))
    return True, relevance_score, cleaned_clinical_notes, "Thank you! Your clinical correction has been recorded to improve accuracy."


def save_training_feedback(
    age: int,
    gender: str,
    features: List[float],
    predicted_dosha: str,
    corrected_dosha: str,
    corrected_colour: str,
    corrected_coating: str,
    corrected_fissures: str,
    raw_notes: str
) -> Dict[str, Any]:
    init_db()
    is_accepted, rel_score, cleaned_notes, reason = sanitize_and_validate_feedback(
        features, corrected_dosha, corrected_colour, corrected_coating, corrected_fissures, raw_notes
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_feedback (
            created_at, patient_age, patient_gender, features_json,
            predicted_dosha, corrected_dosha, corrected_colour,
            corrected_coating, corrected_fissures, raw_notes,
            cleaned_notes, relevance_score, is_accepted, rejection_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(age),
            gender,
            json.dumps(features),
            predicted_dosha,
            corrected_dosha,
            corrected_colour,
            corrected_coating,
            corrected_fissures,
            raw_notes,
            cleaned_notes,
            rel_score,
            1 if is_accepted else 0,
            reason
        )
    )
    conn.commit()
    conn.close()

    return {
        "accepted": is_accepted,
        "relevance_score": rel_score,
        "cleaned_notes": cleaned_notes,
        "message": reason
    }


def predict_with_feedback_calibration(features: List[float], rule_dosha: str, rule_scores: Dict[str, float]) -> Dict[str, Any]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT features_json, corrected_dosha, relevance_score FROM training_feedback WHERE is_accepted = 1")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 2:
        return {
            "calibrated_dosha": rule_dosha,
            "calibrated_scores": rule_scores,
            "accepted_samples_used": len(rows)
        }

    X, y_dosha, weights = [], [], []
    for f_json, c_dosha, rel in rows:
        try:
            vec = json.loads(f_json)
            if len(vec) == len(features):
                X.append(vec)
                y_dosha.append(c_dosha)
                weights.append(float(rel))
        except Exception:
            continue

    if len(X) < 2:
        return {
            "calibrated_dosha": rule_dosha,
            "calibrated_scores": rule_scores,
            "accepted_samples_used": len(X)
        }

    X_arr = np.array(X, dtype=np.float64)
    q_arr = np.array(features, dtype=np.float64)
    std = np.std(X_arr, axis=0) + 1e-4
    dists = np.linalg.norm((X_arr - q_arr) / std, axis=1)

    k = min(5, len(X))
    nearest_idx = np.argsort(dists)[:k]

    knn_votes = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}
    for idx in nearest_idx:
        sim = 1.0 / (1.0 + float(dists[idx]))
        w = sim * weights[idx]
        label = y_dosha[idx]
        if "Vata" in label:
            knn_votes["Vata"] += w
        if "Pitta" in label:
            knn_votes["Pitta"] += w
        if "Kapha" in label:
            knn_votes["Kapha"] += w
        if "Balanced" in label:
            knn_votes["Vata"] += w * 0.33
            knn_votes["Pitta"] += w * 0.33
            knn_votes["Kapha"] += w * 0.33

    tot_vote = sum(knn_votes.values())
    if tot_vote > 1e-5:
        alpha = min(0.30, 0.06 * len(X))
        blended = {}
        for d in ["Vata", "Pitta", "Kapha"]:
            knn_pct = (knn_votes[d] / tot_vote) * 100.0
            blended[d] = round((1.0 - alpha) * rule_scores[d] + alpha * knn_pct, 1)
        s_tot = max(1.0, sum(blended.values()))
        for d in blended:
            blended[d] = round((blended[d] / s_tot) * 100.0, 1)

        top_d = max(blended.items(), key=lambda x: x[1])[0]
        calibrated_dosha = f"{top_d} Predominant" if blended[top_d] >= 40.0 else rule_dosha
        return {
            "calibrated_dosha": calibrated_dosha,
            "calibrated_scores": blended,
            "accepted_samples_used": len(X)
        }

    return {
        "calibrated_dosha": rule_dosha,
        "calibrated_scores": rule_scores,
        "accepted_samples_used": len(X)
    }


def save_patient_review(name: str, age: int, gender: str, rating: int, dominant_dosha: str, review_text: str) -> bool:
    init_db()
    clean_text = re.sub(r"https?://\S+", "", (review_text or "").strip())
    if not clean_text:
        return False
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patient_reviews (created_at, patient_name, age, gender, rating, dominant_dosha, review_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            name.strip()[:60],
            int(age),
            gender,
            int(np.clip(rating, 1, 5)),
            dominant_dosha,
            clean_text[:500]
        )
    )
    conn.commit()
    conn.close()
    return True


def get_patient_reviews(limit: int = 12) -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT created_at, patient_name, age, gender, rating, dominant_dosha, review_text "
        "FROM patient_reviews ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "created_at": r[0],
            "name": r[1],
            "age": r[2],
            "gender": r[3],
            "rating": r[4],
            "dominant_dosha": r[5],
            "review_text": r[6]
        })
    return results
