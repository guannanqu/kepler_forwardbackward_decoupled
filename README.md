# kepler_forwardbackward_decoupled

Quick PyTorch experiments for testing "frozen activation region" ideas against standard neural network baselines.

## Layout

- `datasets.py`: synthetic tabular and image datasets
- `models.py`: standard and frozen-gate MLP/CNN models
- `trainers.py`: shared training and evaluation utilities
- `mlp_test.py`: tabular MLP experiment
- `cnn_test.py`: image CNN experiment
- `online_test.py`: continual binary-task experiment on MNIST/CIFAR-10
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

Online binary-task sequence comparison:

```bash
python test.py online --dataset mnist --num-tasks 5 --epochs-per-task 3 --num-seeds 3
```

You can also run the scripts directly:

```bash
python mlp_test.py --dataset spiral
python cnn_test.py --dataset relative_position
python cnn_test.py --dataset mnist --n-train 5000 --n-test 1000 --noise 0.0
python cnn_test.py --dataset cifar10 --n-train 10000 --n-test 2000 --noise 0.0
python cnn_test.py --dataset relative_position --pooling avg --residual
python cnn_test.py --dataset relative_position --convs-per-stage 3
python online_test.py --dataset cifar10 --num-tasks 5 --epochs-per-task 2
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
- `cnn`: `--channels C1 C2`, `--convs-per-stage`, `--image-size`, `--n-train`, `--n-test`, `--pooling`, `--residual`
- `cnn`: `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-name-prefix`

## W&B note

W&B logging is optional and off by default. If you pass `--wandb`, the code expects the `wandb` package to be installed:

```bash
pip install wandb
```

When enabled, W&B logging now includes:

- per-epoch `train_loss`, `train_acc`, and `valid_acc` for each `(model type, seed)` run
- an additional aggregate run with mean/std trajectories across seeds for both `standard` and `frozen_gate`

## Online task sequence

`online_test.py` turns MNIST or CIFAR-10 into a sequence of binary tasks. For task `k`, the model predicts whether an image belongs to class `k` or not. The same model is trained task by task in sequence, and the script reports:

- mean current-task accuracy across the task sequence
- final average accuracy over all seen tasks
- per-task current accuracy and average-seen-task accuracy, averaged across seeds

If you pass `--wandb` to `online_test.py`, each `(model type, seed)` is logged as one continuous W&B run across the whole task sequence. The logged `global_epoch` keeps increasing across task changes, so with `10` tasks and `10` epochs per task you get one 100-epoch curve rather than separate runs per task.
