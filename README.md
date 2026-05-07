# GNN Over-Squashing Analysis

Empirical study of the **over-squashing** phenomenon in Graph Neural Networks using the _Tree Neighbour-Matching_ benchmark. Four models are compared across binary trees of increasing depth: vanilla **GCN** and **GIN**, and their **Virtual Node** variants (**GCN-VN**, **GIN-VN**).

## Task

Each graph is a complete binary tree. A key-value pair is hidden at one leaf; the root must identify which leaf holds a specific value. Deeper trees require information to travel further, directly stressing over-squashing.


Plots are saved to `results/`: accuracy vs. depth, training time, and per-epoch learning curves.

## Setup

```bash
pip install -r requirements.txt
```

If you want to run the code using uv, install it with `pip install uv` and then run:

```bash
uv sync
uv run -m main
```

> PyTorch must be installed first. For CPU: `pip install torch>=2.1.0`. For CUDA see the [PyTorch install guide](https://pytorch.org/get-started/locally/).


## Running

```bash
python -m main
```

Results and plots are written to `results/`. Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--depths` | `2 3 4 5` | Tree depths to evaluate |
| `--epochs` | `300` | Training epochs per run |
| `--hidden_dim` | `64` | Hidden layer width |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `64` | Batch size |
| `--results_dir` | `results` | Output directory |
| `--plot-only` | `False` | Skip training, regenerate plots only |

**Example — quick run on shallow trees:**

```bash
python -m main --depths 2 3 --epochs 150
```

**Example — regenerate plots from existing CSVs:**

```bash
python -m main --plot-only True
```
