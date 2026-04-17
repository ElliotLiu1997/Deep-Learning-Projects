# Generative Modeling on PathMNIST: WGAN-GP and Diffusion Models

This project implements and evaluates generative models on the PathMNIST dataset, including a Wasserstein GAN with Gradient Penalty (WGAN-GP) and diffusion-based models (DDPM and DDIM). The goal is to compare model performance in terms of image quality, diversity, and sampling efficiency.


## 📁 Project Structure
```
├── dataset.py # Data loading and preprocessing
├── models.py # Generator and Critic architectures
├── train.py # Training pipeline for WGAN-GP
├── metrics.py # Evaluation metrics (FID, IS, Precision, Recall)
└── utils.py # Visualization and utility functions
```

## ⚙️ Requirements

The project requires the following Python packages:

- torch
- torchvision
- numpy
- scipy
- tqdm
- matplotlib

Install them using:

```{bash}
pip install torch torchvision numpy scipy tqdm matplotlib
```

## 📊 Dataset

We use the **PathMNIST** dataset from the MedMNIST collection:

https://medmnist.com/

- RGB images  
- Resolution: 28 × 28  
- Stored in `.npy` format  
- Preprocessed to range `[-1, 1]`  


## 🚀 Training

To train the WGAN-GP model:
```{python}
python train.py --epochs 150 --batch_size 256 --device cuda --gpu_ids 0,1
```
Or run in bash
```{bash}
bash run.sh
```

### Key Training Settings

- Latent dimension: 100  
- Optimizer: Adam  
- Learning rate: 1e-4  
- n_critic: 5  
- Gradient penalty: λ = 5  

## 📈 Evaluation

The model is evaluated using:

- FID (Fréchet Inception Distance)  
- Inception Score (IS)  
- Precision / Recall  
- Sampling Time  

## 🖼️ Outputs

The training pipeline generates:

- Generated image samples (4×4 grids)  
- Real vs generated comparisons  
- Latent space interpolation  
- Training loss curves  
- Model checkpoints  

## 🔍 Summary

- WGAN-GP provides fast sampling with low computational cost  
- Diffusion models achieve better image quality and diversity  
- There is a trade-off between efficiency and generation quality  

## 📌 Notes

- Multi-GPU training is supported via DataParallel  
- Mixed precision training (AMP) is enabled for GPU acceleration  
- The pipeline is designed for reproducibility and modular experimentation  