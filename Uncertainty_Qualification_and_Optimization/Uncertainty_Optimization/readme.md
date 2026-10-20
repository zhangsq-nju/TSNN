# Uncertainty optimization

Evaluates the regularization-based uncertainty optimization on SNN dropout, SNN dropconnect, and SLAYER. Each script adds Gaussian noise of increasing variance to the weights of a trained model, quantizes it with and without the regularization term, and plots the output variance of both.

## SNN dropout and SNN dropconnect

Environment: [`requirements.txt`](../../requirements.txt). The scripts read the four MNIST IDX gzip files from `--data_dir` (default `MNIST`) and load a full-precision model saved with `torch.save(model)`.

```bash
python SNN_dropout_UO.py --model_path path/to/model.pth
python SNN_dropconnect_UO.py --model_path path/to/model.pth
```

| Argument | Default | Description |
| --- | --- | --- |
| `--model_path` | required | Trained full-precision model |
| `--batch_size` | `256` | Batch size |
| `--lamda` | `10` | Weight of the regularization term |
| `--plot` / `--no-plot` | `--plot` | Save `SNN_dropout_curves.pdf` / `SNN_dropconnect_curves.pdf` |
| `--data_dir` | `MNIST` | Directory with the MNIST gzip files |
| `--output_dir` | `.` | Directory for the plot |

## SLAYER

Environment: the [SLAYER build](../../Ternary_Quantization_SNN/Slayer/README.md). `Slayer_UO.py` loads `MLP_regularization_direct_SNN.pth` and `MLP_direct_SNN.pth`, which are produced by `nmnistMLP_regular_direct.py` and `nmnistMLP_direct.py`. Run it from the SLAYER MLP folder, where these weights, `network.yaml` and `NMNISTsmall.zip` are located:

```bash
cd ../../Ternary_Quantization_SNN/Slayer/example/02_NMNIST_MLP
python ../../../../Uncertainty_Qualification_and_Optimization/Uncertainty_Optimization/Slayer_UO.py
```

The plot is saved as `Slayer_curves.pdf`.
