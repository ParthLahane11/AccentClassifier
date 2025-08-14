import os
import pandas as pd
import torch
import soundfile as sf
import numpy as np
from transformers import TrainingArguments
import transformers
import sys
from sklearn.utils.multiclass import unique_labels

# Set backend BEFORE importing Dataset or Audio
from datasets import config
config.AUDIO_BACKEND = "soundfile"

from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, classification_report

# Root directory for audio files
DATASET_ROOT = "./cv-corpus-22.0-delta-2025-06-20/en"

# Load metadata
df = pd.read_csv("metadata.csv")

# Ensure all paths end with .wav
assert df["path"].str.endswith(".wav").all(), "Error: Some files are not .wav"

# Debug prints
print("Unique accents in metadata.csv:", df['accent'].unique())
print("Number of unique accents:", len(df['accent'].unique()))

label2id = {label: i for i, label in enumerate(sorted(df["accent"].unique()))}
print("Label2ID dict:", label2id)
print("Number of labels for training:", len(label2id))
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["accent"].map(label2id)

# Prepend dataset root to path
df["path"] = df["path"].apply(lambda p: os.path.join(DATASET_ROOT, p))

# Create Hugging Face dataset
dataset = Dataset.from_pandas(df[["path", "label"]])

# Train-test split
dataset = dataset.train_test_split(test_size=0.2)
train_ds = dataset["train"]
val_ds = dataset["test"]

# Load processor
MODEL_NAME = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Create fresh config with correct label info to avoid mismatch
config = Wav2Vec2Config.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    mask_time_prob=0.05,
)

# Initialize model from config (not from pretrained weights to avoid label mismatch)
model = Wav2Vec2ForSequenceClassification(config)
model.gradient_checkpointing_enable()

TARGET_SAMPLE_RATE = 16000
FIXED_DURATION_SAMPLES = TARGET_SAMPLE_RATE * 3  # 48000 samples = 3 seconds


def preprocess(batch):
    audio_arrays = []
    labels = []
    for path, label in zip(batch["path"], batch["label"]):
        full_path = os.path.normpath(path)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
        try:
            audio, sr = sf.read(full_path)
            if sr != TARGET_SAMPLE_RATE:
                print(f"Skipping {full_path}: expected 16kHz, got {sr}")
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Enforce fixed length
            if len(audio) > FIXED_DURATION_SAMPLES:
                audio = audio[:FIXED_DURATION_SAMPLES]
            elif len(audio) < FIXED_DURATION_SAMPLES:
                padding = np.zeros(FIXED_DURATION_SAMPLES - len(audio))
                audio = np.concatenate([audio, padding])

            audio_arrays.append(audio)
            labels.append(label)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

    if len(audio_arrays) == 0:
        return {}

    inputs = processor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "input_values": [x for x in inputs["input_values"]],
        "attention_mask": [x for x in inputs["attention_mask"]],
        "label": labels,
    }


print("Preprocessing training data...")
train_ds = train_ds.map(preprocess, batched=True, batch_size=8, remove_columns=["path"])

print("Preprocessing validation data...")
val_ds = val_ds.map(preprocess, batched=True, batch_size=8, remove_columns=["path"])

# Filter out any empty examples
train_ds = train_ds.filter(lambda x: len(x["input_values"]) > 0)
val_ds = val_ds.filter(lambda x: len(x["input_values"]) > 0)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_steps=500,
    logging_steps=10,
    logging_dir="./logs",
    report_to="none",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

print("\nFinal classification report:")
preds = trainer.predict(val_ds)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(axis=1)


all_labels = list(label2id.values())
target_names = [id2label[i] for i in all_labels]

print(classification_report(
    y_true, y_pred,
    labels=all_labels,
    target_names=target_names
))
