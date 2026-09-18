"""
download_data.py -- Download the NSL-KDD dataset into data/raw/.

Usage:
    python src/download_data.py

What it does:
  1. Tries to download KDDTrain+.txt and KDDTest+.txt from multiple mirrors
  2. If ALL downloads fail, generates synthetic fallback data so you can
     keep building without getting stuck

The NSL-KDD dataset is public-domain research data from the University of
New Brunswick (UNB).  It contains ~125K training and ~22K test records of
simulated network traffic labelled as normal or one of four attack categories.
"""

import sys
import os
import urllib.request
import ssl
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR, NSL_KDD_COLUMNS, ATTACK_MAP

# ============================================================
# Download mirrors (tried in order)
# ============================================================
MIRRORS = [
    # Mirror 1: GitHub raw (jmnwong's widely-used mirror)
    {
        "train": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
        "test":  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    },
    # Mirror 2: Another GitHub mirror
    {
        "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    },
]


def _download_file(url: str, dest: Path) -> bool:
    """Download a single file from *url* to *dest*.  Returns True on success."""
    try:
        print(f"  Trying: {url}")
        # Create an SSL context that doesn't verify (some campus networks
        # have MITM proxies that break certificate chains)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            data = resp.read()

        dest.write_bytes(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"  [OK] Saved {dest.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_nslkdd() -> bool:
    """Try each mirror until both files are downloaded.  Returns True on success."""
    train_dest = RAW_DATA_DIR / "KDDTrain+.txt"
    test_dest  = RAW_DATA_DIR / "KDDTest+.txt"

    # Skip if already present
    if train_dest.exists() and test_dest.exists():
        print(f"[OK] Dataset already exists in {RAW_DATA_DIR}")
        return True

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, mirror in enumerate(MIRRORS, 1):
        print(f"\n--- Mirror {i}/{len(MIRRORS)} ---")
        train_ok = _download_file(mirror["train"], train_dest)
        test_ok  = _download_file(mirror["test"], test_dest)
        if train_ok and test_ok:
            return True
        # Clean up partial downloads before trying next mirror
        if train_dest.exists() and not train_ok:
            train_dest.unlink()
        if test_dest.exists() and not test_ok:
            test_dest.unlink()

    return False


# ============================================================
# Fallback: synthetic data generator
# ============================================================
def generate_synthetic_data() -> None:
    """
    Generate a small synthetic dataset that mimics NSL-KDD structure.

    This is ONLY for development/testing when the real dataset cannot be
    downloaded.  The synthetic data has the same columns and label
    distribution but random feature values -- ML results will be poor,
    which is expected.
    """
    import random

    print("\n[FALLBACK] Generating synthetic NSL-KDD-like data...")
    random.seed(42)

    # Use the column names from config (minus difficulty_level for generation,
    # we'll add it back as a dummy column)
    feature_cols = NSL_KDD_COLUMNS[:-2]  # everything except label & difficulty
    categorical = {"protocol_type": ["tcp", "udp", "icmp"],
                   "service": ["http", "smtp", "ftp", "ssh", "dns",
                               "telnet", "private", "other"],
                   "flag": ["SF", "S0", "REJ", "RSTR", "SH",
                            "RSTO", "S1", "S2", "S3", "OTH"]}

    # Pick representative labels from each coarse class
    labels_pool = {
        "Normal": ["normal"] * 50,          # 50% normal
        "DoS":    ["neptune", "smurf", "back"] * 10,
        "Probe":  ["satan", "ipsweep", "portsweep", "nmap"] * 5,
        "R2L":    ["guess_passwd", "ftp_write", "warezmaster"] * 3,
        "U2R":    ["buffer_overflow", "rootkit"] * 2,
    }
    all_labels = []
    for lbls in labels_pool.values():
        all_labels.extend(lbls)

    def _make_row() -> str:
        """Generate one CSV row."""
        parts = []
        for col in feature_cols:
            if col in categorical:
                parts.append(random.choice(categorical[col]))
            elif col in ("land", "logged_in", "is_host_login",
                         "is_guest_login", "root_shell", "su_attempted"):
                parts.append(str(random.randint(0, 1)))
            elif "rate" in col:
                parts.append(f"{random.random():.2f}")
            elif "count" in col or col == "duration":
                parts.append(str(random.randint(0, 500)))
            else:
                parts.append(str(random.randint(0, 65535)))
        label = random.choice(all_labels)
        difficulty = str(random.randint(1, 21))
        parts.extend([label, difficulty])
        return ",".join(parts)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Training set: 20,000 rows
    train_path = RAW_DATA_DIR / "KDDTrain+.txt"
    with open(train_path, "w") as f:
        for _ in range(20_000):
            f.write(_make_row() + "\n")
    print(f"  [OK] Synthetic training data: {train_path}  (20,000 rows)")

    # Test set: 5,000 rows
    test_path = RAW_DATA_DIR / "KDDTest+.txt"
    with open(test_path, "w") as f:
        for _ in range(5_000):
            f.write(_make_row() + "\n")
    print(f"  [OK] Synthetic test data: {test_path}  (5,000 rows)")
    print()
    print("  NOTE: This is synthetic data for development only.")
    print("  ML accuracy will be low.  Replace with real NSL-KDD when possible.")


# ============================================================
# Main
# ============================================================
def main() -> None:
    """Download NSL-KDD or generate synthetic fallback."""
    print("=" * 60)
    print("  NSL-KDD Dataset Download")
    print("=" * 60)

    success = download_nslkdd()

    if success:
        print("\n[OK] Dataset ready!")
    else:
        print("\n[WARNING] All download mirrors failed.")
        print("Generating synthetic fallback data instead...")
        generate_synthetic_data()

    # Quick sanity check: count lines
    for name in ["KDDTrain+.txt", "KDDTest+.txt"]:
        path = RAW_DATA_DIR / name
        if path.exists():
            n_lines = sum(1 for _ in open(path))
            print(f"  {name}: {n_lines:,} rows")
        else:
            print(f"  [ERROR] {name} not found!")


if __name__ == "__main__":
    main()
