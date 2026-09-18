"""
config.py — Central configuration for Self-Evolving IDS.

All paths, constants, hyperparameters, and feature definitions live here.
No other file should hard-code paths or magic numbers.
"""

import os
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================
# Root of the project (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# Model artefacts
MODELS_DIR      = PROJECT_ROOT / "models"

# Database
DB_PATH         = PROJECT_ROOT / "ids_database.sqlite"

# Logs
LOG_DIR         = PROJECT_ROOT / "logs"

# ============================================================
# 2. NSL-KDD DATASET
# ============================================================
# Column names for NSL-KDD (the raw CSVs have no headers)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",            # attack type or "normal"
    "difficulty_level", # NSL-KDD specific score (we drop this)
]

# Categorical feature names (will be one-hot / label encoded)
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# The label column name after loading
LABEL_COLUMN = "label"

# ============================================================
# 3. ATTACK TAXONOMY  (NSL-KDD)
#    Maps fine-grained labels → 5 coarse classes
# ============================================================
ATTACK_MAP = {
    # Normal
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS",
    "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L  (Remote to Local)
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "worm": "R2L",
    # U2R  (User to Root)
    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "rootkit": "U2R", "perl": "U2R", "sqlattack": "U2R",
    "xterm": "U2R", "ps": "U2R",
}

# Which coarse class to hold out for zero-day simulation
ZERO_DAY_CLASS = "R2L"

# ============================================================
# 4. PREPROCESSING HYPERPARAMETERS
# ============================================================
TEST_SIZE      = 0.2     # Fraction for test split
RANDOM_STATE   = 42      # Reproducibility seed
SCALER_PATH    = MODELS_DIR / "scaler.pkl"

# ============================================================
# 5. KNOWN-ATTACK CLASSIFIER (XGBoost)
# ============================================================
KNOWN_MODEL_PATH    = MODELS_DIR / "known_classifier.pkl"
XGBOOST_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "use_label_encoder": False,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
    "verbosity":        1,
}
CONFIDENCE_THRESHOLD = 0.7   # Below this → send to anomaly detector

# ============================================================
# 6. ANOMALY DETECTOR (Autoencoder)
# ============================================================
ANOMALY_MODEL_PATH   = MODELS_DIR / "anomaly_autoencoder.pt"
THRESHOLD_PATH       = MODELS_DIR / "threshold.json"
AE_HIDDEN_DIMS       = [64, 32, 16]   # Encoder layers
AE_EPOCHS            = 50
AE_BATCH_SIZE        = 256
AE_LEARNING_RATE     = 1e-3
AE_THRESHOLD_PERCENTILE = 95          # percentile of normal errors

# ============================================================
# 7. CONTINUAL LEARNING
# ============================================================
REPLAY_BUFFER_SIZE   = 5000           # Max samples in buffer
REPLAY_RATIO         = 0.5            # 50% old, 50% new in fine-tune batch
MODEL_VERSION_DIR    = MODELS_DIR / "versions"

# ============================================================
# 8. DRIFT DETECTION  (ADWIN)
# ============================================================
ADWIN_DELTA          = 0.002          # ADWIN sensitivity

# ============================================================
# 9. EXPLAINABILITY
# ============================================================
SHAP_TOP_K           = 5              # Top-K features to show

# ============================================================
# 10. GUI
# ============================================================
TRAFFIC_FEED_DELAY_MS = 500           # Milliseconds between simulated rows
APP_TITLE = "Self-Evolving IDS — Intrusion Detection System"

# ============================================================
# 11. ENSURE DIRECTORIES EXIST
# ============================================================

def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR,
              MODEL_VERSION_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Auto-create on import so no script ever fails due to missing dirs
ensure_dirs()
