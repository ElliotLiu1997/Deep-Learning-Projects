"""
Project configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parent
    data_root: Path = project_root.parent

    # current workspace layout:
    # /Project2/Flicker8k_Dataset
    # /Project2/Flickr8k_text
    images_dir: Path = data_root / "Flicker8k_Dataset"
    captions_file: Path = data_root / "Flickr8k_text" / "Flickr8k.token.txt"

    train_split_file: Path = data_root / "Flickr8k_text" / "Flickr_8k.trainImages.txt"
    val_split_file: Path = data_root / "Flickr8k_text" / "Flickr_8k.devImages.txt"
    test_split_file: Path = data_root / "Flickr8k_text" / "Flickr_8k.testImages.txt"

    cache_dir: Path = project_root / "cache"
    checkpoints_dir: Path = project_root / "checkpoints"
    outputs_dir: Path = project_root / "outputs"

    features_train_path: Path = cache_dir / "features_train.npy"
    features_val_path: Path = cache_dir / "features_val.npy"
    features_test_path: Path = cache_dir / "features_test.npy"

    feature_keys_train_path: Path = cache_dir / "feature_keys_train.json"
    feature_keys_val_path: Path = cache_dir / "feature_keys_val.json"
    feature_keys_test_path: Path = cache_dir / "feature_keys_test.json"

    metrics_csv_path: Path = outputs_dir / "metrics.csv"
    captions_json_path: Path = outputs_dir / "captions.json"
    training_loss_path: Path = outputs_dir / "training_loss.npy"

    vocab_path: Path = cache_dir / "vocab.json"

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Data preprocessing
    image_size: int = 224
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)

    min_word_freq: int = 1
    max_caption_length: int = 30  # includes <start> and <end>

    # Model
    cnn_feature_dim: int = 512  # ResNet18 pooled output dim
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2  # used for LSTM+Dropout variant

    model_variants: tuple = (
        "rnn",
        "gru",
        "lstm",
        "lstm_dropout",
    )

    # Training
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 3
    num_workers: int = 4
    pin_memory: bool = True

    use_feature_cache: bool = True
    teacher_forcing_ratio: float = 1.0 
    seed: int = 123

    # Decoding  evaluation
    max_decode_length: int = 25
    bleu_n_grams: tuple = (1, 2, 3, 4)


def get_config() -> Config:
    """Create config and ensure required directories exist."""
    cfg = Config()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
