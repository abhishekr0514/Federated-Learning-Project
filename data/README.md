# MedShield-FL Data Storage

This directory manages dataset structures and manifests for MedShield-FL.

## Data Isolation Notice
- **`data/raw/`**: Local raw dataset files (e.g. PTB-XL, Cardiovascular `cardio_train.csv`, NBME text notes). Raw data stored here remains strictly inside local institutional boundaries and is ignored by Git (`.gitignore`).
- **`data/processed/`**: Processed tensor features and local cached index arrays.
- **`data/manifests/`**: Patient-independent split manifests and metadata index files.

Do NOT check in raw patient data, PII, or raw ECG waveforms to version control.
