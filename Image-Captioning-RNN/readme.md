# Image Captioning with CNN-RNN Architectures

This work implements an image captioning system using a CNN-RNN framework on the Flickr8k dataset. Different recurrent models (RNN, GRU, LSTM) are compared using BLEU scores.

---

## 📁 Code Structure

- `config.py`  
  Configuration of paths, hyperparameters, and experiment settings.

- `dataset.py`  
  Dataset and dataloader for Flickr8k, including caption preprocessing.

- `model.py`  
  Model definitions, including the CNN encoder (ResNet-18) and RNN-based decoders.

- `train.py`  
  Training script for all model variants. Saves the best checkpoint based on validation loss.

- `evaluate.py`  
  Evaluation script. Generates captions and computes BLEU scores.

- `feature_cache.py`  
  Extracts and caches CNN features to accelerate training.

- `vocab.py`  
  Builds vocabulary and handles tokenization.

- `utils.py`  
  Utility functions (random seed, checkpoint saving/loading).

- `figure.py`  
  Generates plots for training/validation loss and BLEU score comparisons.

- `run.sh`  
  Shell script to run the full pipeline (training + evaluation + visualization).

---

## 🧰 Requirements
- Python 3.10
- GPU is recommended for faster training

Install dependencies using:

```bash
pip install torch torchvision numpy tqdm nltk pillow matplotlib
```

## 📦 Dataset
We use the Flickr8k dataset:

https://www.kaggle.com/datasets/adityajn105/flickr8k

## 🚀 How to Run

```bash
nohup bash run.sh > run.out 2>&1 &
```

## What This Script Does

Running the pipeline will:

- Build vocabulary (if not already created)
- Generate feature cache (if needed)
- Train all models (RNN, GRU, LSTM, LSTM+Dropout)
- Save best checkpoints
- Evaluate models and compute BLEU scores
- Generate result visualizations

---

## 📊 Outputs

All results are saved in the `outputs/` directory:

- `metrics.csv` — BLEU scores for all models  
- `captions.json` — generated captions and references  
- `training_loss.npy` — training and validation loss history  
- Figures summarize results

---

## 🧠 Notes

- Encoder: pretrained ResNet-18 (frozen)  
- Decoder: RNN / GRU / LSTM variants  
- Evaluation: BLEU scores with multiple references  