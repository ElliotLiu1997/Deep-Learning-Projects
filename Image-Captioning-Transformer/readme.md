# Image Captioning with RNN and Transformer Models

This project implements an image captioning system on the Flickr8k dataset using both recurrent (RNN/GRU/LSTM) and Transformer-based architectures. The goal is to compare model performance and analyze the impact of different visual encoders and decoding mechanisms.

---

## 📁 Project Structure

```
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset and dataloader
├── vocab.py               # Vocabulary and tokenization
├── model.py               # CNN, ViT, RNN, and Transformer models
├── train.py               # Training pipeline
├── evaluate.py            # Caption generation and BLEU evaluation
├── feature_cache.py       # Precompute CNN features (for RNN models)
├── utils.py               # Utility functions
├── figure.py              # Plot training curves and BLEU results
├── outputs/               # Results and figures
```

---

## 📊 Dataset

We use the Flickr8k dataset

https://www.kaggle.com/datasets/adityajn105/flickr8k

---

## ⚙️ Requirements

* Python 3.10
* PyTorch
* torchvision
* numpy
* matplotlib
* nltk
* tqdm
* Pillow

Install dependencies:

```
pip install torch torchvision numpy matplotlib nltk tqdm pillow
```

GPU is recommended for faster training.

---

## 🚀 How to Run

```bash
nohup bash run.sh > run.out 2>&1 &
```


## 📈 Outputs

All outputs are saved in the `outputs/` directory:

* `metrics.csv` — BLEU scores for all models
* `captions.json` — Generated captions and references
* `training_loss.npy` — Training and validation loss history
* `figures/` — Generated plots

---

## 🧠 Model Overview

We implement and compare the following models:

* **RNN-based models**: RNN, GRU, LSTM, LSTM with dropout
* **Transformer-based models**:

  * CNN + Transformer Decoder
  * ViT + Transformer Decoder

Transformer models use:

* Self-attention for modeling caption dependencies
* Cross-attention for aligning text with image features

---

## 📝 Notes

* CNN feature caching is only used for RNN-based models.
* Transformer models operate directly on spatial image features.
* Greedy decoding is used during inference.
* BLEU scores are computed at the corpus level.

---
