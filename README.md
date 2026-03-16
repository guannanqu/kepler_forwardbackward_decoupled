# kepler_forwardbackward_decoupled

Quick PyTorch experiments for testing "frozen activation region" ideas against standard neural network baselines.

## Layout

- `datasets.py`: synthetic tabular and image datasets
- `models.py`: standard and frozen-gate MLP/CNN models
- `trainers.py`: shared training and evaluation utilities
- `mlp_test.py`: tabular MLP experiment
- `cnn_test.py`: image CNN experiment
- `test.py`: small dispatcher for either experiment

## Frozen-gate idea

For the MLP:

- Standard hidden layer: `h = ReLU(Wx + b)`
- Frozen-gate hidden layer: `h = (W x + b) * 1[(W0 x + b0) > 0]`

For the CNN:

- ReLU masks are chosen from the initialization-time conv activations.
- Max-pool winner indices are also chosen from the initialization-time activations.
- The current convolutions and classifier still train normally.

## Run

MLP comparison:

```bash
python test.py mlp --epochs 300 --num-seeds 5
```

CNN comparison on an image task:

```bash
python test.py cnn --epochs 80 --num-seeds 5
```

You can also run the scripts directly:

```bash
python mlp_test.py --dataset spiral
python cnn_test.py --dataset relative_position
python cnn_test.py --dataset mnist --n-train 5000 --n-test 1000 --noise 0.0
python cnn_test.py --dataset cifar10 --n-train 10000 --n-test 2000 --noise 0.0
python cnn_test.py --dataset relative_position --pooling avg --residual
python test.py cnn --wandb --wandb-project kepler-forwardbackward-decoupled
```

## Built-in datasets

Tabular:

- `two_moons`
- `checkerboard`
- `spiral`

Image:

- `patch_xor`: each image contains a left vertical patch and a right horizontal patch; the label depends on whether their top/bottom placements disagree
- `relative_position`: each image contains one blob on the left and one on the right; the label depends on whether the left blob is above the right blob
- `mnist`: handwritten digit classification via `torchvision.datasets.MNIST`
- `cifar10`: 10-class natural image classification via `torchvision.datasets.CIFAR10`

## Useful flags

- `--noise` changes dataset difficulty
- `--cpu` forces CPU even if CUDA is available
- `--wandb` logs each model/seed run to Weights & Biases
- `mlp`: `--depth`, `--width`
- `mlp`: `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-name-prefix`
- `cnn`: `--channels C1 C2`, `--image-size`, `--n-train`, `--n-test`, `--pooling`, `--residual`
- `cnn`: `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-name-prefix`

## W&B note

W&B logging is optional and off by default. If you pass `--wandb`, the code expects the `wandb` package to be installed:

```bash
pip install wandb
```

When enabled, W&B logging now includes:

- per-epoch `train_loss`, `train_acc`, and `valid_acc` for each `(model type, seed)` run
- an additional aggregate run with mean/std trajectories across seeds for both `standard` and `frozen_gate`
