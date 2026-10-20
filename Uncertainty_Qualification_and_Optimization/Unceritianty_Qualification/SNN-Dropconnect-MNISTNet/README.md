# SNN-Dropconnect-MNISTNet

Uncertainty quantification on MNIST (downloaded to `./data`). Environment: [`requirements.txt`](../../../requirements.txt).

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/FCSNN_dropout.pth` |
| Direct quantization | `nets/FCSNN_dropout_direct.pth` |
| Indirect quantization | `nets/FCSNN_dropout_indirect2.pth` |

Each file must contain a complete model object saved with `torch.save(model)`.

```bash
python UncertaintyQualification.py
```
