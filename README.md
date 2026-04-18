# Deception Detection using EEG Signals: A Comparative Analysis of Feature Engineering and Deep Learning Model
---
## 📌 Project Overview

This project investigates whether deceptive behavior can be automatically detected 
from non-invasive EEG signals using deep learning. It frames lie detection as a 
binary classification task (Truth vs. Lie) using short EEG windows recorded from 
27 participants via a wearable Emotiv Insight headset.

---

## 🧠 Dataset

- **Name:** LieWaves EEG Dataset  
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/5gzxb2bzs2/1)  
- **Participants:** 27  
- **Channels:** AF3, AF4, T7, T8, Pz (5 channels)  
- **Sampling Rate:** 128 Hz  
- **Task:** Modified Concealed Information Test (CIT)

---

## ⚙️ Methods & Pipeline

### Preprocessing
- Band-pass filtering (0.5–40 Hz)
- ATAR artifact removal
- Z-score normalization per segment
- Sliding window segmentation (128 samples, 1 second)

### Feature Extraction
| Method | Description |
|--------|-------------|
| Raw EEG | Filtered time-series fed directly to CNN-LSTM |
| FFT | Power Spectral Density across Theta, Alpha, Beta bands |
| DWT | Wavelet energy with time-frequency localization (db4, Level 5) |
| FFT + DWT Fusion | Combined 30-feature vector |

### Models
- **CNN-LSTM** — baseline on filtered EEG windows
- **Dense MLP** — trained on FFT, DWT, and fused features

### Evaluation
- Subject-wise train/test split (cross-subject generalization)
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📊 Results Summary

| Approach                | Test Accuracy | F1 Score |
|-------------------------|---------------|----------|
| CNN-LSTM (Bandpass)     | 54.4%         | 0.617    |
| CNN-LSTM (ATAR)         | 53.9%         | 0.521    |
| FFT Features (MLP)      | 54.9%         | 0.527    |
| DWT Features (MLP)      | 50.6%         | 0.564    |
| FFT + DWT Fusion (MLP)  | 54.0%         | 0.537    |

> Best accuracy: ~56% under subject-wise evaluation — slightly above chance,  
> reflecting the inherent difficulty of cross-subject EEG lie detection.

---

## 🛠️ Tech Stack

- Python, Jupyter Notebook
- TensorFlow / Keras
- NumPy, SciPy, scikit-learn
- PyWavelets (DWT)
- Matplotlib, Seaborn

---
