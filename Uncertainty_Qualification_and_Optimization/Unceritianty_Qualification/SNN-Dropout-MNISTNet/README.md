# SNN-Dropout-MNISTNet

Uncertainty quantification on MNIST (downloaded to `./data`). Environment: [`requirements.txt`](../../../requirements.txt).

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/FCSNN_dropout.pth` |
| Direct quantization | `nets/FCSNN_direct_dropout.pth` |
| Indirect quantization | `nets/FCSNN_indirect_dropout2.pth` |

Each file must contain a `state_dict` of the network defined in this folder.

```bash
python UncertaintyQualification.py
```
