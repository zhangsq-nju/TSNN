# SNN-PLIF-MNISTNet

Uncertainty quantification on MNIST (downloaded to `./data`). Environment: [PLIF `requirements.txt`](../../../Ternary_Quantization_SNN/PLIF/requirements.txt).

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/plif.pt` |
| Direct quantization | `nets/plif_direct.pt` |
| Indirect quantization | `nets/plif_indirect.pt` |

Each file must contain a PLIF training checkpoint (a dictionary with the model under `net`, as saved by the PLIF scripts).

```bash
python UncertaintyQualification.py
```
