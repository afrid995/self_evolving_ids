"""
preprocess.py -- Load, clean, encode, scale, and split the NSL-KDD dataset.

Usage:
    python src/preprocess.py

What it does:
  1. Loads KDDTrain+.txt and KDDTest+.txt from data/raw/
     (runs download_data.py automatically if files are missing)
  2. Assigns column names, drops the difficulty_level column
  3. Maps fine-grained attack labels -> 5 coarse classes
  4. Holds out the zero-day class (R2L) for later simulation
  5. Encodes categorical features (LabelEncoder)
  6. Scales numerical features (StandardScaler)
  7. Splits training data into train/validation sets
  8. Saves everything to data/processed/ as Parquet files
  9. Saves the scaler and label encoders to models/

Output files in data/processed/:
  - X_train.parquet, y_train.parquet
  - X_val.parquet,   y_val.parquet
  - X_test.parquet,  y_test.parquet
  - X_zeroday.parquet, y_zeroday.parquet   (held-out R2L class)
  - feature_names.txt                       (ordered feature list)
  - label_classes.txt                       (ordered class names)

Output files in models/:
  - scaler.pkl         (fitted StandardScaler)
  - label_encoders.pkl (dict of fitted LabelEncoders for categoricals)
  - label_mapping.pkl  (LabelEncoder for the target column)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR,
    NSL_KDD_COLUMNS, CATEGORICAL_FEATURES, LABEL_COLUMN,
    ATTACK_MAP, ZERO_DAY_CLASS,
    TEST_SIZE, RANDOM_STATE, SCALER_PATH,
)


# ============================================================
# 1. Loading
# ============================================================
def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load a raw NSL-KDD text file into a DataFrame.

    The files are CSV-like with no header row.  We assign column names
    from config.NSL_KDD_COLUMNS and drop the 'difficulty_level' column.
    """
    print(f"  Loading {filepath.name}...")
    df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
    # Drop the NSL-KDD-specific difficulty score (not a network feature)
    df.drop(columns=["difficulty_level"], inplace=True)
    print(f"    Rows: {len(df):,}  Columns: {len(df.columns)}")
    return df


# ============================================================
# 2. Label mapping
# ============================================================
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map fine-grained attack names to coarse classes using ATTACK_MAP.

    Unknown labels (attacks in test set not seen in training) are mapped
    to 'Unknown' -- these will be treated as potential zero-day traffic.
    """
    print("  Mapping labels to coarse classes...")
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.lower()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(ATTACK_MAP).fillna("Unknown")

    # Show distribution
    counts = df[LABEL_COLUMN].value_counts()
    for cls, cnt in counts.items():
        pct = 100 * cnt / len(df)
        print(f"    {cls:10s}: {cnt:>7,}  ({pct:5.1f}%)")

    return df


# ============================================================
# 3. Zero-day holdout
# ============================================================
def split_zeroday(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate the zero-day class from the main dataset.

    Returns:
        main_df:    All rows that are NOT the zero-day class
        zeroday_df: All rows of the zero-day class
    """
    print(f"  Holding out zero-day class: '{ZERO_DAY_CLASS}'...")
    mask = df[LABEL_COLUMN] == ZERO_DAY_CLASS
    zeroday_df = df[mask].copy()
    main_df = df[~mask].copy()
    print(f"    Main dataset:  {len(main_df):,} rows")
    print(f"    Zero-day held: {len(zeroday_df):,} rows")
    return main_df, zeroday_df


# ============================================================
# 4. Encoding categorical features
# ============================================================
def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns in-place.

    Args:
        df:       DataFrame to encode
        encoders: Pre-fitted encoders (used for test/zeroday sets)
        fit:      If True, fit new encoders; if False, use existing ones

    Returns:
        Encoded DataFrame and the dict of encoders
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            # Fit on this column; add '<unknown>' for unseen values later
            le.fit(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]

        # Transform, handling unseen values gracefully
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x, _k=known, _le=le: (
                _le.transform([x])[0] if x in _k else -1
            )
        )

    return df, encoders


# ============================================================
# 5. Encode target labels
# ============================================================
def encode_target(
    series: pd.Series,
    le: LabelEncoder | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode the target column to integer labels.

    Returns the encoded array and the fitted LabelEncoder.
    """
    if fit:
        le = LabelEncoder()
        le.fit(series)
    encoded = le.transform(series)
    return encoded, le


# ============================================================
# 6. Scale numerical features
# ============================================================
def scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScaler on all columns (categoricals are already integers).

    Returns the scaled DataFrame and the fitted scaler.
    """
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        scaler.fit(df)

    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return df, scaler


# ============================================================
# 7. Save helpers
# ============================================================
def save_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    """Save a DataFrame as Parquet and print confirmation."""
    filepath = path / f"{name}.parquet"
    df.to_parquet(filepath, index=False)
    print(f"    Saved {name}.parquet  ({len(df):,} rows)")


def save_text(items: list[str], path: Path, name: str) -> None:
    """Save a list of strings as a text file (one per line)."""
    filepath = path / f"{name}.txt"
    filepath.write_text("\n".join(items))
    print(f"    Saved {name}.txt  ({len(items)} items)")


# ============================================================
# 8. Main pipeline
# ============================================================
def run_preprocessing() -> dict:
    """
    Execute the full preprocessing pipeline.

    Returns a dict with keys: X_train, y_train, X_val, y_val,
    X_test, y_test, X_zeroday, y_zeroday, feature_names, label_classes
    """
    print("=" * 60)
    print("  Phase 1: Preprocessing NSL-KDD")
    print("=" * 60)

    # --- Check if raw data exists; download if not ---
    train_path = RAW_DATA_DIR / "KDDTrain+.txt"
    test_path  = RAW_DATA_DIR / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        print("\n[INFO] Raw data not found. Running download script...")
        from src.download_data import main as download_main
        download_main()

    if not train_path.exists() or not test_path.exists():
        print("[ERROR] Data files still missing after download attempt!")
        print("  Please manually place KDDTrain+.txt and KDDTest+.txt in:")
        print(f"  {RAW_DATA_DIR}")
        sys.exit(1)

    # --- Step 1: Load ---
    print("\n[Step 1/7] Loading raw data...")
    train_df = load_raw_data(train_path)
    test_df  = load_raw_data(test_path)

    # --- Step 2: Map labels ---
    print("\n[Step 2/7] Mapping attack labels...")
    print("  Training set:")
    train_df = map_labels(train_df)
    print("  Test set:")
    test_df  = map_labels(test_df)

    # --- Step 3: Hold out zero-day class ---
    print("\n[Step 3/7] Splitting zero-day holdout...")
    print("  From training set:")
    train_df, zeroday_train = split_zeroday(train_df)
    print("  From test set:")
    test_df, zeroday_test = split_zeroday(test_df)

    # Combine zero-day from both sets
    zeroday_df = pd.concat([zeroday_train, zeroday_test], ignore_index=True)
    print(f"  Total zero-day samples: {len(zeroday_df):,}")

    # --- Step 4: Encode categoricals ---
    print("\n[Step 4/7] Encoding categorical features...")
    # Fit on training data only
    train_df, cat_encoders = encode_categoricals(train_df, fit=True)
    # Transform test and zero-day using same encoders
    test_df, _    = encode_categoricals(test_df, encoders=cat_encoders, fit=False)
    zeroday_df, _ = encode_categoricals(zeroday_df, encoders=cat_encoders, fit=False)
    print(f"    Encoded columns: {CATEGORICAL_FEATURES}")

    # --- Step 5: Separate features and target ---
    print("\n[Step 5/7] Separating features and target...")
    feature_cols = [c for c in train_df.columns if c != LABEL_COLUMN]

    X_train_raw = train_df[feature_cols]
    y_train_raw = train_df[LABEL_COLUMN]
    X_test      = test_df[feature_cols]
    y_test_raw  = test_df[LABEL_COLUMN]
    X_zeroday   = zeroday_df[feature_cols]
    y_zeroday_raw = zeroday_df[LABEL_COLUMN]

    # Encode target labels — fit on ALL coarse classes (including zero-day)
    # so the encoder can handle zero-day labels when we use them later.
    all_classes = sorted(set(y_train_raw) | set(y_test_raw) | set(y_zeroday_raw))
    target_le = LabelEncoder()
    target_le.fit(all_classes)

    y_train_enc   = target_le.transform(y_train_raw)
    y_test_enc    = target_le.transform(y_test_raw)
    y_zeroday_enc = target_le.transform(y_zeroday_raw)

    label_classes = list(target_le.classes_)
    print(f"    Classes: {label_classes}")
    print(f"    Features: {len(feature_cols)}")

    # --- Step 6: Scale features ---
    print("\n[Step 6/7] Scaling features (StandardScaler)...")
    X_train_scaled, scaler = scale_features(X_train_raw, fit=True)
    X_test_scaled, _       = scale_features(X_test, scaler=scaler, fit=False)
    X_zeroday_scaled, _    = scale_features(X_zeroday, scaler=scaler, fit=False)

    # Train/validation split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_enc,
    )
    print(f"    Train: {len(X_train_final):,}  Val: {len(X_val):,}")

    # --- Step 7: Save everything ---
    print("\n[Step 7/7] Saving processed data...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Features
    save_parquet(X_train_final.reset_index(drop=True), PROCESSED_DIR, "X_train")
    save_parquet(X_val.reset_index(drop=True), PROCESSED_DIR, "X_val")
    save_parquet(pd.DataFrame(X_test_scaled).reset_index(drop=True), PROCESSED_DIR, "X_test")
    save_parquet(pd.DataFrame(X_zeroday_scaled).reset_index(drop=True), PROCESSED_DIR, "X_zeroday")

    # Targets (as single-column DataFrames)
    save_parquet(pd.DataFrame({"label": y_train_final}), PROCESSED_DIR, "y_train")
    save_parquet(pd.DataFrame({"label": y_val}), PROCESSED_DIR, "y_val")
    save_parquet(pd.DataFrame({"label": y_test_enc}), PROCESSED_DIR, "y_test")
    save_parquet(pd.DataFrame({"label": y_zeroday_enc}), PROCESSED_DIR, "y_zeroday")

    # Metadata
    save_text(feature_cols, PROCESSED_DIR, "feature_names")
    save_text(label_classes, PROCESSED_DIR, "label_classes")

    # Scaler and encoders
    joblib.dump(scaler, SCALER_PATH)
    print(f"    Saved scaler -> {SCALER_PATH.name}")
    joblib.dump(cat_encoders, MODELS_DIR / "label_encoders.pkl")
    print(f"    Saved label_encoders.pkl")
    joblib.dump(target_le, MODELS_DIR / "label_mapping.pkl")
    print(f"    Saved label_mapping.pkl")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)
    print(f"  Train:    {len(X_train_final):,} samples")
    print(f"  Val:      {len(X_val):,} samples")
    print(f"  Test:     {len(X_test_scaled):,} samples")
    print(f"  Zero-day: {len(X_zeroday_scaled):,} samples (held out '{ZERO_DAY_CLASS}')")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes:  {label_classes}")
    print(f"\n  All files saved to: {PROCESSED_DIR}")
    print(f"  Scaler/encoders in: {MODELS_DIR}")

    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test_enc,
        "X_zeroday": X_zeroday_scaled,
        "y_zeroday": y_zeroday_enc,
        "feature_names": feature_cols,
        "label_classes": label_classes,
    }


if __name__ == "__main__":
    run_preprocessing()
