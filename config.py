from dataclasses import dataclass, field

import torch


@dataclass
class ExperimentConfig:
    # Tree depths to sweep; each depth doubles the number of leaves
    depths: list[int] = field(default_factory=lambda: [2, 3, 4, 5])

    # Hidden dimension shared by every layer of every model
    hidden_dim: int = 64

    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 64
    # Fraction of generated graphs used for training (rest → test)
    train_fraction: float = 0.8
    # Master seed applied before every (model, depth) run for reproducibility
    seed: int = 42

    results_dir: str = "results"

    plot_only: bool = (
        False  # If True, skip training and only generate plots from existing CSVs in results_dir
    )

    # Auto-detected at instantiation; override with ExperimentConfig(device=torch.device("cpu"))
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
