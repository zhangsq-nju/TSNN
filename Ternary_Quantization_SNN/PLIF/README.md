# PLIF

[中文](README_cn.md) · [Original PLIF README](README.upstream.md)

Ternary quantization of PLIF networks from *[Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks](https://arxiv.org/abs/2007.05785)*. The model definitions and baseline training scripts come from the [original PLIF repository](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron).

## Files

| File | Description |
| --- | --- |
| `codes/models.py` | Network definitions and data transforms |
| `codes/train.py` | Full-precision training (from the original repository) |
| `codes/train_val.py` | Full-precision training with a validation split (from the original repository) |
| `codes/direct.py` | Direct ternary quantization of a pretrained full-precision model |
| `codes/indirect.py` | Indirect ternary quantization |
| `codes/indirectKD.py` | Indirect quantization with knowledge distillation (MNIST, Fashion-MNIST, CIFAR-10) |

## Requirements

The code uses the 2020 SpikingJelly API (`spikingjelly.clock_driven`), so it needs an older PyTorch than the other experiments:

```bash
pip install -r requirements.txt
```

This installs SpikingJelly at commit `73f94ab983d0167623015537f7d4460b064cfca1`, the version used by the original PLIF code.

## Usage

Run from this directory. Datasets are read from `-dataset_dir/<dataset_name>` (default `./data`), and logs and checkpoints are written under `-log_dir_prefix` (default `./logs`).

```bash
# Full-precision baseline
python codes/train.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100

# Direct quantization, starting from the baseline checkpoint
python codes/direct.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100 \
  -checkpoint logs/pt_CIFAR10_init_tau_2.0_use_plif_True_use_max_pool_True_T_8_detach_reset_True/check_point_max.pt

# Indirect quantization
python codes/indirect.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100

# Indirect quantization with KD, quantizing the fully connected layers
python codes/indirectKD.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name MNIST -T 5 -max_epoch 200 -quantize -KD -portion fc
```

`-portion` selects the quantized layers in `indirectKD.py` (`all`, `fc`, or `conv`). Use different `-log_dir_prefix` values for different methods, because each script resumes from `check_point.pt` when it finds one in its output folder.
