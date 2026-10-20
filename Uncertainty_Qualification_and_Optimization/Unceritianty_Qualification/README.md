# Uncertainty quantification

Compares the uncertainty of full-precision, directly quantized, and indirectly quantized SNNs. Each folder holds one model/dataset pair, with its own network definition and `UncertaintyQualification.py`.

| Folder | Dataset | Environment |
| --- | --- | --- |
| [SNN-Dropconnect-CIFARNet](SNN-Dropconnect-CIFARNet/README.md) | CIFAR-10 | [`requirements.txt`](../../requirements.txt) |
| [SNN-Dropconnect-MNISTNet](SNN-Dropconnect-MNISTNet/README.md) | MNIST | [`requirements.txt`](../../requirements.txt) |
| [SNN-Dropout-CIFARNet](SNN-Dropout-CIFARNet/README.md) | CIFAR-10 | [`requirements.txt`](../../requirements.txt) |
| [SNN-Dropout-MNISTNet](SNN-Dropout-MNISTNet/README.md) | MNIST | [`requirements.txt`](../../requirements.txt) |
| [SNN-PLIF-MNISTNet](SNN-PLIF-MNISTNet/README.md) | MNIST | [PLIF `requirements.txt`](../../Ternary_Quantization_SNN/PLIF/requirements.txt) |
| [SNN-Slayer-MNISTNet](SNN-Slayer-MNISTNet/README.md) | N-MNIST | SLAYER (built from this folder) |

## Usage

1. Train the full-precision, direct, and indirect models with the scripts in [`Ternary_Quantization_SNN`](../../Ternary_Quantization_SNN).
2. Put the three checkpoints in the folder's `nets/` directory, using the file names listed in its README.
3. Run the analysis from that folder:

```bash
cd SNN-Dropout-CIFARNet
python UncertaintyQualification.py
```

MNIST and CIFAR-10 are downloaded through torchvision. The results are plotted with Matplotlib.
