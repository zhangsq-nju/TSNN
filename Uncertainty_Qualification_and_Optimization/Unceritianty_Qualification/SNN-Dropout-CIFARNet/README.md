# SNN-Dropout-CIFARNet

Uncertainty quantification on CIFAR-10 (downloaded to `./data/cifar`). Environment: [`requirements.txt`](../../../requirements.txt).

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/CSNN_dropout.pth` |
| Direct quantization | `nets/CSNN_dropout_direct2.pth` |
| Indirect quantization | `nets/CSNN_dropout_indirect2.pth` |

Each file must contain a complete model object saved with `torch.save(model)`.

```bash
python UncertaintyQualification.py
```
