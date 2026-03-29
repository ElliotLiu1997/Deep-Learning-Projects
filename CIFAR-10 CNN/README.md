# CIFAR-10 CNN Architecture Comparison Project

This project implements and compares six convolutional neural network (CNN) architectures on the CIFAR-10 dataset.  
The study evaluates the impact of:

- Network depth
- Residual connections
- Batch normalization

All models are trained under identical optimization and data settings to ensure fair comparison.

# 1. Project Structure

```
.
├── main.py                # Entry point for training and evaluation
├── models.py              # CNN model definitions
├── train.py               # Training and evaluation logic
├── utils.py               # Data loading, plotting, reproducibility utilities
├── run_parallel.sh        # Script to train all 6 models in parallel
└── README.md
```

# 2. Requirements

## Required Packages

Python 3.10.

Install dependencies using:

```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```

GPU is recommended.

# 3. Dataset

This project uses the CIFAR-10 dataset.

Download from:

https://www.cs.toronto.edu/~kriz/cifar.html

After downloading, place the dataset in your desired data directory.

The script assumes the dataset is already downloaded.  
Set the dataset path using the `--data_dir` argument when running.

# 4. How to Run (Single Model)

To train and evaluate a single model:

```bash
python main.py \
  --data_dir <path_to_cifar10> \
  --output_dir outputs \
  --epochs 35 \
  --batch_size 128 \
  --num_workers 4 \
  --lr 0.001 \
  --seed 123 \
  --model baseline
```

Available model options:

- baseline
- deeper
- residual
- baseline_bn
- deeper_bn
- residual_bn
- all  (runs all models sequentially)

Example:

```bash
python main.py --data_dir .. --model residual_bn
```

# 5. Run All Models in Parallel (Multi-GPU)

To train all six models using multiple GPUs example:

```bash
./run_parallel.sh .. ../outputs_parallel 25 128 4 0.001 42
```

## Script Behavior

- Uses GPU 0 and GPU 1 (My environment)
- Trains up to two models simultaneously and automatically assigns models to available GPUs
- Logs are saved to:

```
<output_dir>/logs/
```

Each model runs independently and saves its outputs.

# 6. Output Files

For each model, the following files are generated:

```
outputs/<model_name>/
├── best_<model_name>.pt
├── train_loss_curve.png
├── val_accuracy_curve.png
├── confusion_matrix.png
├── training_history.csv
├── test_predictions.csv
```

Descriptions:

- `best_*.pt` — best model checkpoint based on validation accuracy
- `train_loss_curve.png` — training/validation loss plot
- `val_accuracy_curve.png` — training/validation accuracy plot
- `confusion_matrix.png` — confusion matrix visualization
- `training_history.csv` — per-epoch metrics
- `test_predictions.csv` — test predictions

# 7. Training Configuration

Default training settings:

- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 5e-4
- Scheduler: StepLR (step_size=10, gamma=0.1)
- Batch size: 128
- Epochs: 25
- Loss function: CrossEntropyLoss

All models are trained using identical settings for fair comparison.

# 8. Expected Results

Best-performing model:

Residual CNN + Batch Normalization

Approximate test accuracy:

~87%

Minor variations may occur depending on hardware and random seed.

---