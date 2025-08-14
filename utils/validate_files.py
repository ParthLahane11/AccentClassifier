import os
import soundfile as sf
import pandas as pd

DATASET_ROOT = "../cv-corpus-22.0-delta-2025-06-20/en"

df = pd.read_csv("../metadata.csv")

def clean_path(p):
    return os.path.normpath(os.path.join(DATASET_ROOT, p))

df["file_path"] = df["path"].apply(clean_path)

print("Checking all audio files for existence and readability...")

errors = []

for idx, path in enumerate(df["file_path"]):
    if not os.path.exists(path):
        print(f"[MISSING] File does not exist: {path}")
        errors.append((path, "Missing file"))
        continue

    try:
        data, sr = sf.read(path)
        if sr != 16000:
            print(f"[WARNING] Sampling rate != 16000 Hz for: {path} (got {sr})")
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        errors.append((path, str(e)))

print(f"\nChecked {len(df)} files. Found {len(errors)} problem files.")

if errors:
    print("Problem files:")
    for file_path, err_msg in errors:
        print(f" - {file_path}: {err_msg}")
else:
    print("All files exist and are readable.")
