# SNN with dropout

Ternary quantization of SNNs with dropout: a fully connected network on MNIST and a convolutional network on CIFAR-10.

| Method | MNIST | CIFAR-10 |
| --- | --- | --- |
| Direct | `MNIST_direct.py` | `CIFAR_direct.py` |
| Indirect | `MNIST_indirect.py` | `CIFAR_indirect.py` |
| Indirect + KD | `MNIST_indirectKD.py` | `CIFAR_indirectKD.py` |

## Requirements

```bash
pip install -r ../../requirements.txt
```

## Data

CIFAR-10 is downloaded automatically. The MNIST scripts read the four IDX gzip files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`) and do not download them.

| Script | Default `--data_dir` |
| --- | --- |
| `MNIST_direct.py`, `MNIST_indirect.py` | `MNIST` |
| `MNIST_indirectKD.py` | `data/MNIST` |
| `CIFAR_direct.py`, `CIFAR_indirect.py` | `./data` |
| `CIFAR_indirectKD.py` | `data/CIFAR10/` |

## Usage

Run from this directory:

```bash
python MNIST_direct.py --epochs 100 --batch_size 128 --learning_rate 1e-4 -T 10
python MNIST_indirect.py --epochs 99 --batch_size 128 --learning_rate 1e-4 -T 10 --qn 100
python MNIST_indirectKD.py --epochs 200 --batch_size 256 --learning_rate 1e-4 -T 5
```

The CIFAR-10 scripts take the same arguments.

| Argument | Description |
| --- | --- |
| `--epochs` | Training epochs (per quantization round for the indirect scripts) |
| `--batch_size`, `--learning_rate` | Batch size and learning rate |
| `-T` | Number of simulation time steps |
| `--qn` | Indirect scripts: number of train-then-quantize rounds |
| `--data_dir` | Dataset directory (see above) |
| `--output_dir` | Parent directory for W&B logs, TensorBoard logs and checkpoints (default `.`) |
| `--checkpoint_dir` | KD scripts: checkpoint folder under `--output_dir` (default `checkpoints`) |

The direct and indirect scripts log accuracy to W&B; set `WANDB_MODE=offline` to keep the logs local. The KD scripts write TensorBoard logs to `runs/` and save checkpoints as `state_dict`.
