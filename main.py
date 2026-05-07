import argparse
import csv
import os
import random
import time
from dataclasses import replace

import numpy as np
import torch

from config import ExperimentConfig
from data.tree_dataset import TreeDataset
from models.baselines import GCN, GIN
from models.virtual_node import GCN_VN, GIN_VN
from utils.plotting import generate_all_plots


def set_seed(seed: int) -> None:
    """Fix all random sources for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One full pass over the training set.

    Returns:
        avg_loss (float), accuracy (float in [0, 1])
    """
    model.train()
    total_loss = correct = total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.root_mask, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader (no gradient computation).

    Returns:
        avg_loss (float), accuracy (float in [0, 1])
    """
    model.eval()
    total_loss = correct = total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.root_mask, batch.batch)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def run_experiment(
    model_class, model_name: str, depth: int, cfg: ExperimentConfig
) -> dict:
    """Train one model on one tree depth from scratch.

    num_layers is set to depth + 1: one layer to encode leaf key-value pairs
    before aggregation, then depth hops to move leaf information to the root.

    Returns a dict with:
        - scalar summary metrics  (best_test_acc, training_time_s, …)
        - full epoch history lists (train_acc_history, test_acc_history)
          for later use by the Step 3 plotting module
    """
    set_seed(cfg.seed)

    dataset = TreeDataset(depth=depth)
    train_loader, test_loader, in_dim, out_dim, criterion = dataset.generate_data(
        train_fraction=cfg.train_fraction,
        batch_size=cfg.batch_size,
    )

    model = model_class(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        num_layers=depth + 1,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    best_test_acc = 0.0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        _, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device
        )
        _, test_acc = evaluate(model, test_loader, criterion, cfg.device)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 50 == 0:
            print(
                f"    [{model_name} | depth={depth}] "
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}"
            )

    elapsed = time.time() - t_start

    return {
        "model": model_name,
        "depth": depth,
        "best_test_acc": best_test_acc,
        "final_train_acc": train_acc_history[-1],
        "final_test_acc": test_acc_history[-1],
        "training_time_s": round(elapsed, 2),
        # Full histories are consumed by utils/plotting.py in Step 3
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history,
    }


def save_summary_csv(results: list[dict], path: str) -> None:
    """Write one row per (model, depth) run to a CSV file."""
    fieldnames = [
        "model",
        "depth",
        "best_test_acc",
        "final_train_acc",
        "final_test_acc",
        "training_time_s",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})


def save_epoch_csv(results: list[dict], path: str) -> None:
    """Write one row per epoch per (model, depth) run — used by Step 3 plots."""
    fieldnames = ["model", "depth", "epoch", "train_acc", "test_acc"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            for epoch, (tr, te) in enumerate(
                zip(r["train_acc_history"], r["test_acc_history"]), start=1
            ):
                writer.writerow(
                    {
                        "model": r["model"],
                        "depth": r["depth"],
                        "epoch": epoch,
                        "train_acc": tr,
                        "test_acc": te,
                    }
                )


def parse_args(cfg: ExperimentConfig) -> ExperimentConfig:
    """Parse command-line arguments to override default config values."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=cfg.epochs, help="Number of training epochs"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=cfg.hidden_dim, help="Hidden dimension size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=cfg.batch_size, help="Training batch size"
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=cfg.depths,
        help="Tree depths to experiment with",
    )
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Learning rate")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=cfg.train_fraction,
        help="Training set fraction",
    )
    parser.add_argument(
        "--seed", type=int, default=cfg.seed, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=cfg.results_dir,
        help="Directory to save results CSVs and plots",
    )
    parser.add_argument(
        "--plot-only",
        type=bool,
        default=False,
        help="Skip training and only generate plots from existing CSVs in results_dir",
    )

    args = parser.parse_args()

    return replace(cfg, **vars(args))


def main() -> None:
    cfg = ExperimentConfig()
    cfg = parse_args(cfg)

    if not cfg.plot_only:
        os.makedirs(cfg.results_dir, exist_ok=True)
        print(f"Device : {cfg.device}")
        print(f"Depths : {cfg.depths}")
        print(
            f"Epochs : {cfg.epochs}  |  hidden_dim={cfg.hidden_dim}  |  lr={cfg.lr}\n"
        )

        baseline_summary_path = os.path.join(cfg.results_dir, "base_models_summary.csv")
        vn_summary_path = os.path.join(
            cfg.results_dir, "virtual_node_models_summary.csv"
        )
        baseline_epoch_path = os.path.join(cfg.results_dir, "base_models_epochs.csv")
        vn_epoch_path = os.path.join(cfg.results_dir, "virtual_node_models_epochs.csv")

        base_models = [(GCN, "GCN"), (GIN, "GIN")]
        vn_models = [(GCN_VN, "GCN_VN"), (GIN_VN, "GIN_VN")]
        models_registry = {"base_models": base_models, "virtual_node_models": vn_models}
        all_results: dict[str, list[dict]] = {
            "base_models": [],
            "virtual_node_models": [],
        }
        csv_paths: dict[str, tuple[str, str]] = {
            "base_models": (baseline_summary_path, baseline_epoch_path),
            "virtual_node_models": (vn_summary_path, vn_epoch_path),
        }

        for model_types, model_list in models_registry.items():
            print(f"\n  {model_types.replace('_', ' ').title()}")
            print(csv_paths[model_types][0])  # summary CSV path
            for depth in cfg.depths:
                print(f"\n{'─'*60}")
                print(
                    f"  Tree depth = {depth}  (nodes={2**(depth+1)-1}, leaves={2**depth})"
                )
                print(f"{'─'*60}")

                for model_class, model_name in model_list:
                    print(f"\n  Training {model_name}  (num_layers={depth}) …")
                    result = run_experiment(model_class, model_name, depth, cfg)
                    all_results[model_types].append(result)
                    print(
                        f"  ✓ best_test_acc={result['best_test_acc']:.4f}  "
                        f"time={result['training_time_s']:.1f}s"
                    )

            summary_path, epoch_path = csv_paths[model_types]

            # save_summary_csv(all_results, summary_path)
            # save_epoch_csv(all_results, epoch_path)

            print(f"\nSummary → {summary_path}")
            print(f"Epochs  → {epoch_path}")

            print(
                f"\n{'Model':<8} {'Depth':>5}  {'Best test acc':>14}  {'Time (s)':>10}"
            )
            print("─" * 44)
            for r in all_results[model_types]:
                print(
                    f"{r['model']:<8} {r['depth']:>5}  "
                    f"{r['best_test_acc']:>14.4f}  "
                    f"{r['training_time_s']:>10.1f}"
                )

    # Generate all publication-ready charts from every CSV present in results/
    generate_all_plots(cfg.results_dir)


if __name__ == "__main__":
    main()
