"""
app.py — Main entry point for Self-Evolving IDS.

Usage:
    python app.py              # Launch PySide6 GUI (default)
    python app.py --tkinter    # Launch Tkinter GUI (Phase 1)

This file will be completed in the GUI phases.
For now it simply verifies that the project skeleton loads correctly.
"""

import sys
from src import config  # noqa: F401  — verifies config loads


def main() -> None:
    """Launch the IDS application."""
    print(f"[OK] Project root : {config.PROJECT_ROOT}")
    print(f"[OK] Data dir     : {config.DATA_DIR}")
    print(f"[OK] Models dir   : {config.MODELS_DIR}")
    print(f"[OK] Database     : {config.DB_PATH}")
    print()
    print("Self-Evolving IDS skeleton loaded successfully!")
    print("Next step -> Phase 1: Preprocess the NSL-KDD dataset.")


if __name__ == "__main__":
    main()
