# Diffusion Model on PathMNIST

This project implements diffusion-based image generation using DDPM and DDIM on the PathMNIST dataset. It includes a full pipeline for training, sampling, and evaluation, along with both quantitative and qualitative analysis.

---

## 📦 Requirements

- Python 3.10
- PyTorch
- torchvision
- numpy
- matplotlib
- scipy
- tqdm

## 📊 Dataset

This project uses the PathMNIST dataset from the MedMNIST https://medmnist.com/.

### Preprocessing

- Images are normalized to [-1, 1]
- Format converted from (H, W, C) → (C, H, W)

---

## 🗂️ Code Structure

    .
    ├── config.py
    ├── dataset.py
    ├── diffusion.py
    ├── model_unet.py
    ├── train.py
    ├── evaluate.py
    ├── utils.py
    ├── save_real_grid.py
    ├── test.py
    ├── run_job.sh


## 🚀 How to Run

Run the entire pipeline using:

```bash
bash run_job.sh 150 256 256 500 0,1
```

## ⚙️ Arguments

    bash run_job.sh <epochs> <batch_size> <eval_batch_size> <num_samples> <gpu_ids>

| Argument         | Description                                   |
|-----------------|-----------------------------------------------|
| epochs          | Training epochs                     |
| batch_size      | Training batch size                           |
| eval_batch_size | Batch size used during evaluation             |
| num_samples     | Number of generated samples    |
| gpu_ids         | Comma-separated GPU IDs (e.g., `0`, `0,1`)    |

## 📤 Outputs

All outputs are saved under:

    outputs/

### Generated files

- `samples/`  
  → Generated images (DDPM, DDIM, per-epoch samples)

- `plots/loss_curve.png`  
  → Training and validation loss curve

- `diffusion_process/`  
  → Diffusion forward/reverse visualization

- `metrics.csv`  
  → Evaluation results:
  - FID
  - Inception Score (IS)
  - Precision / Recall
  - Sampling time

- `latest.pt`  
  → Trained model checkpoint

---

## 📈 Features

- DDPM and DDIM sampling
- Cosine noise schedule
- U-Net with sinusoidal time embedding
- EMA (Exponential Moving Average)
- Multi-GPU training (DataParallel)

### Quantitative evaluation

- FID
- Inception Score (IS)
- Precision / Recall

## 🧪 Additional Scripts

- `test.py`  
  → Verify dataset normalization and tensor ranges  

- `save_real_grid.py`  
  → Save real image samples for comparison  

---

## 📌 Notes

- The model is trained without labels (unconditional generation)
- Generated images are samples from the learned data distribution, not reconstructions
- Some generated samples may contain noise or lack clear structure, especially with fewer DDIM steps
