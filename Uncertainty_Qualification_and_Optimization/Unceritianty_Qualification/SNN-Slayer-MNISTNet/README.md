# SNN-Slayer-MNISTNet

Uncertainty quantification on the N-MNIST subset in `NMNISTsmall/`. Environment: SLAYER, built from this folder.

Place the trained checkpoints in `nets/`:

| Model | File |
| --- | --- |
| Full precision | `nets/MLP_SNN.pth` |
| Direct quantization | `nets/MLP_direct_SNN.pth` |
| Indirect quantization | `nets/MLP_indirect_SNN2.pth` |

Each file must contain a `state_dict` of the MLP (as saved by the SLAYER scripts).

This folder contains its own copy of SLAYER. Build it in a separate environment from `Ternary_Quantization_SNN/Slayer`, because both install the same package names:

```bash
python setup.py install
python UncertaintyQualification.py
```
