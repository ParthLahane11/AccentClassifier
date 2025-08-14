import pandas as pd
import os
from pydub import AudioSegment
from tqdm import tqdm

DATASET_DIR = "../cv-corpus-22.0-delta-2025-06-20/en"
TSV_PATH = os.path.join(DATASET_DIR, "validated.tsv")
CLIPS_DIR = os.path.join(DATASET_DIR, "clips")
OUTPUT_CSV = "../metadata.csv"

WAV_DIR = os.path.join(DATASET_DIR, "clips_wav")
os.makedirs(WAV_DIR, exist_ok=True)

# Parameters for uniform audio length
FIXED_DURATION_MS = 3000  # 3 seconds
TARGET_SAMPLE_RATE = 16000

def map_raw_accent(accent_str):
    if pd.isna(accent_str):
        return None
    accent_str = accent_str.lower()
    if any(x in accent_str for x in ["united states", "general american", "california", "san francisco", "mid atlantic", "midwestern", "alabama", "cuban"]):
        return "american"
    elif any(x in accent_str for x in ["british", "england", "uk", "yorkshire", "cornish", "welsh"]):
        return "british"
    elif "australian" in accent_str:
        return "australian"
    elif "new zealand" in accent_str:
        return "new_zealand"
    elif any(x in accent_str for x in ["south africa", "southern african", "zimbabwe", "namibia"]):
        return "south_african"
    elif any(x in accent_str for x in ["india", "south asia", "sri lanka", "pakistan"]):
        return "indian"
    elif any(x in accent_str for x in ["east african", "kenyan", "ugandan", "eastern africa"]):
        return "east_african"
    elif "filipino" in accent_str:
        return "filipino"
    else:
        return None

print("Loading validated.tsv...")
df = pd.read_csv(TSV_PATH, sep='\t')

df = df[df['accents'].notnull() & df['path'].notnull()]
df['mapped_accent'] = df['accents'].apply(map_raw_accent)
df = df[df['mapped_accent'].notnull()]

print("Converting mp3 to fixed-length wav...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    mp3_path = os.path.join(CLIPS_DIR, row['path'])
    wav_filename = os.path.splitext(row['path'])[0] + ".wav"
    wav_path = os.path.join(WAV_DIR, wav_filename)

    if not os.path.exists(wav_path):
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)

        if len(audio) > FIXED_DURATION_MS:
            audio = audio[:FIXED_DURATION_MS]
        elif len(audio) < FIXED_DURATION_MS:
            padding = AudioSegment.silent(duration=FIXED_DURATION_MS - len(audio))
            audio += padding  # pad at the end

        audio.export(wav_path, format="wav")

    df.at[idx, 'path'] = os.path.relpath(wav_path, DATASET_DIR)

accent_to_label = {accent: idx for idx, accent in enumerate(sorted(df['mapped_accent'].unique()))}
print("Accent to label mapping:", accent_to_label)
print("Number of unique accents:", len(accent_to_label))
df['accent_label'] = df['mapped_accent'].map(accent_to_label)

output_df = df[['path', 'mapped_accent', 'accent_label', 'client_id']]
output_df.columns = ['path', 'accent', 'accent_label', 'client_id']
output_df.to_csv(OUTPUT_CSV, index=False)

print("metadata.csv saved.")
print(f"Labels: {accent_to_label}")
print(f"Total samples: {len(output_df)}")
