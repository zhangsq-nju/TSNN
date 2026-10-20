# PLIF

[English](README.md) · [PLIF 原始 README](README_cn.upstream.md)

对论文 *[Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks](https://arxiv.org/abs/2007.05785)* 中的 PLIF 网络进行三值量化。模型定义和基线训练脚本来自 [PLIF 原始仓库](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)。

## 文件

| 文件 | 说明 |
| --- | --- |
| `codes/models.py` | 网络定义和数据变换 |
| `codes/train.py` | 全精度训练（来自原始仓库） |
| `codes/train_val.py` | 划分验证集的全精度训练（来自原始仓库） |
| `codes/direct.py` | 对预训练全精度模型进行直接三值量化 |
| `codes/indirect.py` | 间接三值量化 |
| `codes/indirectKD.py` | 结合知识蒸馏的间接量化（MNIST、Fashion-MNIST、CIFAR-10） |

## 依赖

代码使用 2020 年版 SpikingJelly 的接口（`spikingjelly.clock_driven`），需要比其他实验更旧的 PyTorch：

```bash
pip install -r requirements.txt
```

该命令会安装 commit 为 `73f94ab983d0167623015537f7d4460b064cfca1` 的 SpikingJelly，即 PLIF 原始代码所用的版本。

## 使用

在本目录下运行。数据集从 `-dataset_dir/<dataset_name>` 读取（默认 `./data`），日志和检查点写入 `-log_dir_prefix`（默认 `./logs`）。

```bash
# 全精度基线
python codes/train.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100

# 直接量化，从基线检查点开始
python codes/direct.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100 \
  -checkpoint logs/pt_CIFAR10_init_tau_2.0_use_plif_True_use_max_pool_True_T_8_detach_reset_True/check_point_max.pt

# 间接量化
python codes/indirect.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name CIFAR10 -T 8 -max_epoch 100

# 结合知识蒸馏的间接量化，只量化全连接层
python codes/indirectKD.py -init_tau 2.0 -use_plif -use_max_pool -detach_reset \
  -device cuda:0 -dataset_name MNIST -T 5 -max_epoch 200 -quantize -KD -portion fc
```

`indirectKD.py` 用 `-portion` 选择量化的层（`all`、`fc` 或 `conv`）。不同方法请使用不同的 `-log_dir_prefix`，因为脚本在输出目录中找到 `check_point.pt` 时会从它继续训练。
