# Audio Classification with PyTorch(Work in Progress)

This repository contains code for training and evaluating an **Audio Classification** model using **PyTorch**.  
It processes `.wav` files, extracts spectrogram features, and trains a neural network to classify audio into different categories.

---

## 📂 Project Structure

```
.
├── data/               # Audio dataset directory
├── models/             # Model definitions and checkpoints
├── utils/              # Utility functions
├── main.py             # Training script
├── inference.py        # Script for running inference
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```
---

## Dataset
The dataset used in this project is **Mozilla Common Voice Corpus 22.0** in English,  
specifically the file:  
`cv-corpus-22.0-delta-2025-06-20-en.tar.gz`  

You can download the dataset from the official Common Voice website:  
[https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)

---
---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/audio-classification.git
cd audio-classification
```

2. **Create and activate a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎙 Dataset

- Audio files should be in `.wav` format.
- Organize them in subfolders, where each folder name corresponds to the class label:
```
data/
│
├── class1/
│   ├── file1.wav
│   ├── file2.wav
│
├── class2/
│   ├── file1.wav
│   ├── file2.wav
│
...
```

---

## 📊 Training the Model

Run the training script:
```bash
python main.py --data_dir data --epochs 20 --batch_size 32
```

**Arguments**:
- `--data_dir` : Path to the dataset folder
- `--epochs` : Number of training epochs
- `--batch_size` : Training batch size
- `--learning_rate` : Learning rate (default: 0.001)

---

## 🔍 Evaluating the Model

After training, run:
```bash
python main.py --evaluate --data_dir data
```

---

## 🎯 Inference

To predict the class of a single `.wav` file:
```bash
python inference.py --file_path path/to/audio.wav
```

---

## ⚠️ Common Issues

- **ValueError: Number of classes does not match size of target_names**  
  This happens when the dataset labels and `target_names` list do not match.  
  ✅ Make sure:
  1. Your dataset has the same number of unique class folders as `target_names`.
  2. You update `target_names` in your code whenever you change the dataset classes.

---

## 📈 Example Output

After training, you might see output like:
```
Epoch 1/20
Train Loss: 0.6543, Accuracy: 78.5%
Validation Loss: 0.5021, Accuracy: 84.2%
```

---

## 🛠 Requirements

See `requirements.txt`. Key libraries include:
- `torch`
- `torchaudio`
- `numpy`
- `scikit-learn`

Install them with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
