# SLAYER

[Original SLAYER README](README.upstream.md) · [License (GPL-3.0)](LICENSE)

Ternary quantization of an N-MNIST MLP trained with [SLAYER](https://github.com/bamsumit/slayerPytorch). This folder bundles the SLAYER framework (`src/`, `test/`, `docs/`, `example/`, `exampleLoihi/`). The quantization experiments are in `example/02_NMNIST_MLP/`.

## Installation

SLAYER compiles C++/CUDA extensions, so it needs a CUDA toolkit matching your PyTorch build. Install PyTorch for your CUDA version, then:

```bash
pip install numpy matplotlib pyyaml h5py tqdm wandb
python setup.py install
```

The CUDA flags in `setup.py` target `-arch=sm_60`; adjust them for your GPU if needed.

## Usage

Run from `example/02_NMNIST_MLP/`. The scripts read `network.yaml` and extract `NMNISTsmall.zip` from this directory, so place the archive there first.

```bash
cd example/02_NMNIST_MLP
python nmnistMLP.py             # full-precision baseline -> MLP_SNN.pth
python nmnistMLP_direct.py      # direct quantization     -> MLP_direct_SNN.pth
python nmnistMLP_indirect.py    # indirect quantization   -> MLP_indirect_SNN.pth
```

| Script | Description |
| --- | --- |
| `nmnistMLP.py` | Full-precision training |
| `nmnistMLP_direct.py`, `nmnistMLP_indirect.py` | Direct / indirect quantization of `MLP_SNN.pth` |
| `direct.py`, `indirect.py` | Same experiments with W&B logging only, without saving weights |
| `nmnistMLP_regular_direct.py` | Direct quantization with the uncertainty regularizer -> `MLP_regularization_direct_SNN.pth` |
| `nmnistMLP_regular_indirect.py` | Compares the output variance of `MLP_regularization_direct_SNN.pth` and `MLP_direct_SNN.pth` under weight noise |
| `certainty.py` | Accuracy of `MLP_SNN.pth` under Gaussian weight noise |
| `MACs.py` | MAC and parameter count of the MLP |

Hyperparameters are set in the scripts and in `network.yaml`.
