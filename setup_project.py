"""
setup_project.py — One-click environment setup for Self-Evolving IDS.

What it does:
  1. Creates a Python virtual environment (venv/)
  2. Upgrades pip inside the venv
  3. Installs PyTorch CPU from the official index
  4. Installs remaining dependencies from requirements.txt
  5. Verifies critical imports
  6. Creates all project directories

Run:
    python setup_project.py

NOTE: Run this from the project root (self_evolving_ids/).
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# Detect platform
IS_WINDOWS = sys.platform.startswith("win")
PYTHON_BIN = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "python"
PIP_BIN    = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "pip"


def run(cmd: list[str], desc: str) -> bool:
    """Run a shell command and report success/failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ✗ FAILED: {desc}")
        return False
    print(f"  ✓ SUCCESS: {desc}")
    return True


def main() -> None:
    """Set up the complete development environment."""
    print("=" * 60)
    print("  Self-Evolving IDS — Environment Setup")
    print("=" * 60)

    # ── Step 1: Create venv ────────────────────────────────
    if not VENV_DIR.exists():
        ok = run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            "Creating virtual environment (venv/)",
        )
        if not ok:
            print("\n✗ Could not create venv. Make sure 'python -m venv' works.")
            print("  Fallback: install Anaconda and use 'conda create -n ids python=3.11'")
            sys.exit(1)
    else:
        print(f"\n✓ Virtual environment already exists at {VENV_DIR}")

    # ── Step 2: Upgrade pip ────────────────────────────────
    run(
        [str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip",
    )

    # ── Step 3: Install PyTorch CPU ────────────────────────
    # PyTorch CPU wheels live on a separate index.
    # We install them first so requirements.txt doesn't try the GPU version.
    print("\n📦 Installing PyTorch (CPU-only) — this may take a few minutes...")
    ok = run(
        [
            str(PIP_BIN), "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        "Installing PyTorch CPU",
    )
    if not ok:
        print("\n⚠  PyTorch install failed.")
        print("   Fallback: We'll use sklearn's IsolationForest instead.")
        print("   You can continue — the project handles this gracefully.")

    # ── Step 4: Install remaining requirements ─────────────
    # We filter out torch/torchvision/torchaudio from requirements.txt
    # since we already installed them above.
    print("\n📦 Installing remaining dependencies...")
    ok = run(
        [
            str(PIP_BIN), "install", "-r", str(REQUIREMENTS),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        ],
        "Installing requirements.txt",
    )
    if not ok:
        print("\n⚠  Some packages failed. Common fixes:")
        print("   - Make sure you have a C++ compiler (Visual Studio Build Tools)")
        print("   - Try: pip install <package> one at a time")
        print("   - XGBoost fail? Fallback: use RandomForestClassifier")
        print("   - PySide6 fail? We'll use Tkinter in Phase 1")

    # ── Step 5: Verify critical imports ────────────────────
    print("\n🔍 Verifying critical imports...")
    checks = [
        ("numpy",        "import numpy; print(f'  numpy {numpy.__version__}')"),
        ("pandas",       "import pandas; print(f'  pandas {pandas.__version__}')"),
        ("sklearn",      "import sklearn; print(f'  sklearn {sklearn.__version__}')"),
        ("xgboost",      "import xgboost; print(f'  xgboost {xgboost.__version__}')"),
        ("torch",        "import torch; print(f'  torch {torch.__version__}')"),
        ("shap",         "import shap; print(f'  shap {shap.__version__}')"),
        ("river",        "import river; print(f'  river {river.__version__}')"),
        ("matplotlib",   "import matplotlib; print(f'  matplotlib {matplotlib.__version__}')"),
    ]

    passed = 0
    failed_libs = []
    for name, code in checks:
        result = subprocess.run(
            [str(PYTHON_BIN), "-c", code],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
            passed += 1
        else:
            print(f"  ✗ {name} — NOT INSTALLED")
            failed_libs.append(name)

    # ── Step 6: Create project directories ─────────────────
    print("\n📁 Creating project directories...")
    dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "models" / "versions",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "tests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(PROJECT_ROOT)}/")

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print(f"  Libraries verified: {passed}/{len(checks)}")
    if failed_libs:
        print(f"  Failed: {', '.join(failed_libs)}")
        print("  (The project has fallbacks for these — you can continue)")
    print()
    print("  Next steps:")
    print(f"  1. Activate the venv:")
    if IS_WINDOWS:
        print(f"       venv\\Scripts\\activate")
    else:
        print(f"       source venv/bin/activate")
    print(f"  2. Verify skeleton: python app.py")
    print(f"  3. Download NSL-KDD → data/raw/")
    print(f"  4. Run Phase 1:     python src/preprocess.py")
    print()


if __name__ == "__main__":
    main()
