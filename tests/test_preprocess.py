"""
test_preprocess.py -- Unit tests for the preprocessing pipeline.

Usage:
    python -m pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PROCESSED_DIR, MODELS_DIR, SCALER_PATH,
    NSL_KDD_COLUMNS, CATEGORICAL_FEATURES,
    LABEL_COLUMN, ZERO_DAY_CLASS,
)
from src.preprocess import (
    load_raw_data,
    map_labels,
    split_zeroday,
    encode_categoricals,
    encode_target,
    scale_features,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small DataFrame mimicking NSL-KDD structure."""
    data = {
        "duration":      [0, 100, 200, 50, 10],
        "protocol_type": ["tcp", "udp", "tcp", "icmp", "tcp"],
        "service":       ["http", "smtp", "ftp", "eco_i", "http"],
        "flag":          ["SF", "S0", "REJ", "SF", "SF"],
        "src_bytes":     [200, 0, 100, 500, 300],
        "dst_bytes":     [1000, 0, 200, 0, 800],
        "land":          [0, 0, 0, 0, 0],
        "wrong_fragment": [0, 0, 1, 0, 0],
        "urgent":        [0, 0, 0, 0, 0],
        "hot":           [0, 0, 0, 0, 0],
        "num_failed_logins": [0, 0, 3, 0, 0],
        "logged_in":     [1, 0, 0, 0, 1],
        "label":         ["normal", "neptune", "guess_passwd",
                          "ipsweep", "buffer_overflow"],
    }
    return pd.DataFrame(data)


# ============================================================
# Tests
# ============================================================
class TestMapLabels:
    """Tests for label mapping."""

    def test_known_labels_mapped(self, sample_df: pd.DataFrame) -> None:
        """Known attack names should map to coarse classes."""
        result = map_labels(sample_df)
        expected = ["Normal", "DoS", "R2L", "Probe", "U2R"]
        assert list(result[LABEL_COLUMN]) == expected

    def test_unknown_label_maps_to_unknown(self) -> None:
        """Labels not in ATTACK_MAP should become 'Unknown'."""
        df = pd.DataFrame({LABEL_COLUMN: ["totally_new_attack"]})
        result = map_labels(df)
        assert result[LABEL_COLUMN].iloc[0] == "Unknown"

    def test_case_insensitive(self) -> None:
        """Mapping should be case-insensitive."""
        df = pd.DataFrame({LABEL_COLUMN: ["Normal", "NORMAL", "nOrMaL"]})
        result = map_labels(df)
        assert all(result[LABEL_COLUMN] == "Normal")


class TestSplitZeroday:
    """Tests for zero-day holdout."""

    def test_zeroday_separated(self, sample_df: pd.DataFrame) -> None:
        """The zero-day class should be removed from main set."""
        mapped = map_labels(sample_df)
        main_df, zeroday_df = split_zeroday(mapped)
        assert ZERO_DAY_CLASS not in main_df[LABEL_COLUMN].values
        assert all(zeroday_df[LABEL_COLUMN] == ZERO_DAY_CLASS)

    def test_row_counts_preserved(self, sample_df: pd.DataFrame) -> None:
        """Total rows = main + zeroday."""
        mapped = map_labels(sample_df)
        main_df, zeroday_df = split_zeroday(mapped)
        assert len(main_df) + len(zeroday_df) == len(mapped)


class TestEncodeCategoricals:
    """Tests for categorical encoding."""

    def test_output_is_numeric(self, sample_df: pd.DataFrame) -> None:
        """Encoded columns should be integers."""
        encoded, encoders = encode_categoricals(sample_df, fit=True)
        for col in CATEGORICAL_FEATURES:
            if col in encoded.columns:
                assert encoded[col].dtype in (np.int64, np.int32, int, object)

    def test_unseen_category_gets_minus_one(self) -> None:
        """Unseen categories at transform time should get -1."""
        train = pd.DataFrame({
            "protocol_type": ["tcp", "udp"],
            "service": ["http", "smtp"],
            "flag": ["SF", "S0"],
        })
        _, encs = encode_categoricals(train, fit=True)
        test = pd.DataFrame({
            "protocol_type": ["icmp"],   # unseen
            "service": ["http"],         # seen
            "flag": ["SF"],              # seen
        })
        result, _ = encode_categoricals(test, encoders=encs, fit=False)
        assert result["protocol_type"].iloc[0] == -1


class TestEncodeTarget:
    """Tests for target encoding."""

    def test_roundtrip(self) -> None:
        """Encoding then decoding should return original labels."""
        labels = pd.Series(["Normal", "DoS", "Probe"])
        encoded, le = encode_target(labels, fit=True)
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == list(labels)


class TestScaleFeatures:
    """Tests for feature scaling."""

    def test_scaled_mean_near_zero(self) -> None:
        """After scaling, mean of each column should be ~0."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        scaled, scaler = scale_features(df, fit=True)
        assert abs(scaled["a"].mean()) < 1e-10
        assert abs(scaled["b"].mean()) < 1e-10

    def test_scaled_std_near_one(self) -> None:
        """After scaling, std of each column should be ~1."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        scaled, _ = scale_features(df, fit=True)
        assert abs(scaled["a"].std(ddof=0) - 1.0) < 0.01


class TestProcessedFilesExist:
    """
    Integration test: verify that processed files exist after
    running the full pipeline.  Skip if preprocessing hasn't
    been run yet.
    """

    @pytest.mark.skipif(
        not (PROCESSED_DIR / "X_train.parquet").exists(),
        reason="Run 'python src/preprocess.py' first",
    )
    def test_all_output_files_exist(self) -> None:
        """All expected output files should exist."""
        expected = [
            "X_train.parquet", "y_train.parquet",
            "X_val.parquet",   "y_val.parquet",
            "X_test.parquet",  "y_test.parquet",
            "X_zeroday.parquet", "y_zeroday.parquet",
            "feature_names.txt", "label_classes.txt",
        ]
        for fname in expected:
            assert (PROCESSED_DIR / fname).exists(), f"Missing: {fname}"

    @pytest.mark.skipif(
        not SCALER_PATH.exists(),
        reason="Run 'python src/preprocess.py' first",
    )
    def test_scaler_saved(self) -> None:
        """Scaler pickle should exist."""
        assert SCALER_PATH.exists()

    @pytest.mark.skipif(
        not (MODELS_DIR / "label_mapping.pkl").exists(),
        reason="Run 'python src/preprocess.py' first",
    )
    def test_label_mapping_saved(self) -> None:
        """Label mapping pickle should exist."""
        assert (MODELS_DIR / "label_mapping.pkl").exists()
