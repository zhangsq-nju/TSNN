# SNN-Dropconnect-CIFARNet

Uncertainty quantification on CIFAR-10 (downloaded to `./data/cifar`). Environment: [`requirements.txt`](../../../requirements.txt).

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/CSNN_dropconnect.pth` |
| Direct quantization | `nets/CSNN_direct_dropconnect.pth` |
| Indirect quantization | `nets/CSNN_indirect_dropconnect.pth` |

Each file must contain a `state_dict` of the network defined in this folder.

```bash
python UncertaintyQualification.py
```
