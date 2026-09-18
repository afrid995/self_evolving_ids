# Self-Evolving Intrusion Detection System

> A desktop application that detects known and zero-day network attacks using
> machine learning, continuously learns from analyst feedback, and monitors
> for concept drift — all with a professional GUI.

## Features

- **Known-attack classification** — XGBoost multi-class classifier
- **Zero-day detection** — PyTorch autoencoder anomaly detector
- **Continual learning** — Replay-buffer fine-tuning without catastrophic forgetting
- **Concept-drift detection** — ADWIN from the River library
- **Explainable AI** — SHAP values for every alert
- **Human-in-the-loop** — Analyst confirms / rejects zero-day alerts
- **Professional GUI** — PySide6 desktop app with live traffic simulation

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd self_evolving_ids

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NSL-KDD dataset into data/raw/ (see below)

# 5. Preprocess
python src/preprocess.py

# 6. Train models
python src/train_known.py
python src/train_anomaly.py

# 7. Launch the app
python app.py
```

## Dataset

This project uses the **NSL-KDD** dataset.
Download from: https://www.unb.ca/cic/datasets/nsl.html

Place these files in `data/raw/`:
- `KDDTrain+.txt`
- `KDDTest+.txt`

## Tech Stack

| Layer          | Tool                     |
|----------------|--------------------------|
| Language       | Python 3.11+             |
| ML             | scikit-learn, XGBoost    |
| Deep Learning  | PyTorch (CPU)            |
| Drift          | River (ADWIN)            |
| Explainability | SHAP                     |
| GUI            | PySide6                  |
| Storage        | SQLite, Parquet          |
| Testing        | pytest                   |

## Project Structure

```
self_evolving_ids/
├── data/raw/            # Original dataset CSVs
├── data/processed/      # Cleaned, split, scaled data
├── src/                 # Core logic modules
├── models/              # Saved model files
├── gui/                 # Desktop GUI (Tkinter + PySide6)
├── tests/               # Unit tests
├── notebooks/           # Jupyter exploration
├── app.py               # Main entry point
└── requirements.txt     # Dependencies
```

## License

This project is for academic/educational purposes.
All dependencies are free and open-source.
