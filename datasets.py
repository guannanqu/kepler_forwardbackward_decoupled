import math

import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


def make_two_moons(n_samples: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = torch.rand(n_outer, generator=generator) * math.pi
    theta_inner = torch.rand(n_inner, generator=generator) * math.pi

    outer = torch.stack([torch.cos(theta_outer), torch.sin(theta_outer)], dim=1)
    inner = torch.stack([1.0 - torch.cos(theta_inner), -torch.sin(theta_inner) - 0.5], dim=1)

    x = torch.cat([outer, inner], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_outer, dtype=torch.long),
            torch.ones(n_inner, dtype=torch.long),
        ]
    )

    x = x + noise * torch.randn(x.shape, generator=generator)
    permutation = torch.randperm(n_samples, generator=generator)
    return x[permutation], y[permutation]


def make_checkerboard(n_samples: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = 4.0 * torch.rand(n_samples, 2, generator=generator) - 2.0
    shifted = x + noise * torch.randn(x.shape, generator=generator)
    cell_x = torch.floor(shifted[:, 0]).to(torch.long)
    cell_y = torch.floor(shifted[:, 1]).to(torch.long)
    y = ((cell_x + cell_y) % 2 != 0).to(torch.long)
    return shifted, y


def make_spiral(n_samples: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n_first = n_samples // 2
    n_second = n_samples - n_first

    radius_first = torch.linspace(0.15, 1.0, n_first)
    radius_second = torch.linspace(0.15, 1.0, n_second)

    theta_first = 4.0 * math.pi * radius_first
    theta_second = theta_first[:n_second] + math.pi

    first = torch.stack(
        [
            radius_first * torch.cos(theta_first),
            radius_first * torch.sin(theta_first),
        ],
        dim=1,
    )
    second = torch.stack(
        [
            radius_second * torch.cos(theta_second),
            radius_second * torch.sin(theta_second),
        ],
        dim=1,
    )

    x = torch.cat([first, second], dim=0)
    x = x + noise * torch.randn(x.shape, generator=generator)
    y = torch.cat(
        [
            torch.zeros(n_first, dtype=torch.long),
            torch.ones(n_second, dtype=torch.long),
        ]
    )
    permutation = torch.randperm(n_samples, generator=generator)
    return x[permutation], y[permutation]


def _draw_patch(image: torch.Tensor, center_y: int, center_x: int, patch: torch.Tensor) -> None:
    patch_h, patch_w = patch.shape
    top = center_y - patch_h // 2
    left = center_x - patch_w // 2
    image[top : top + patch_h, left : left + patch_w] += patch


def make_patch_xor_images(
    n_samples: int,
    noise: float,
    seed: int,
    image_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    vertical_patch = torch.tensor(
        [
            [0.2, 0.9, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.9, 0.2],
        ],
        dtype=torch.float32,
    )
    horizontal_patch = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )

    top_y = image_size // 4
    bottom_y = 3 * image_size // 4
    left_x = image_size // 4
    right_x = 3 * image_size // 4

    for index in range(n_samples):
        left_top = torch.rand((), generator=generator).item() > 0.5
        right_top = torch.rand((), generator=generator).item() > 0.5
        labels[index] = int(left_top != right_top)

        left_y = top_y if left_top else bottom_y
        right_y = top_y if right_top else bottom_y

        jitter = torch.randint(-1, 2, (4,), generator=generator)
        _draw_patch(images[index, 0], left_y + int(jitter[0]), left_x + int(jitter[1]), vertical_patch)
        _draw_patch(images[index, 0], right_y + int(jitter[2]), right_x + int(jitter[3]), horizontal_patch)

        for _ in range(3):
            distractor_y = int(torch.randint(2, image_size - 2, (1,), generator=generator).item())
            distractor_x = int(torch.randint(2, image_size - 2, (1,), generator=generator).item())
            images[index, 0, distractor_y - 1 : distractor_y + 1, distractor_x - 1 : distractor_x + 1] += 0.25

    images += noise * torch.randn(images.shape, generator=generator)
    images = images.clamp(0.0, 1.0)
    permutation = torch.randperm(n_samples, generator=generator)
    return images[permutation], labels[permutation]


def make_relative_position_images(
    n_samples: int,
    noise: float,
    seed: int,
    image_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    blob = torch.tensor(
        [
            [0.1, 0.5, 0.1],
            [0.5, 1.0, 0.5],
            [0.1, 0.5, 0.1],
        ],
        dtype=torch.float32,
    )

    left_x = image_size // 4
    right_x = 3 * image_size // 4
    min_y = 3
    max_y = image_size - 4

    for index in range(n_samples):
        left_y = int(torch.randint(min_y, max_y + 1, (1,), generator=generator).item())
        right_y = int(torch.randint(min_y, max_y + 1, (1,), generator=generator).item())
        labels[index] = int(left_y < right_y)

        x_jitter = torch.randint(-1, 2, (2,), generator=generator)
        _draw_patch(images[index, 0], left_y, left_x + int(x_jitter[0]), blob)
        _draw_patch(images[index, 0], right_y, right_x + int(x_jitter[1]), blob)

        for _ in range(5):
            distractor_y = int(torch.randint(2, image_size - 2, (1,), generator=generator).item())
            distractor_x = int(torch.randint(2, image_size - 2, (1,), generator=generator).item())
            images[index, 0, distractor_y, distractor_x] += 0.35

    images += noise * torch.randn(images.shape, generator=generator)
    images = images.clamp(0.0, 1.0)
    permutation = torch.randperm(n_samples, generator=generator)
    return images[permutation], labels[permutation]


def make_image_dataset(
    name: str,
    n_train: int,
    n_test: int,
    noise: float,
    seed: int,
    image_size: int = 16,
) -> tuple[TensorDataset, TensorDataset]:
    if name in {"mnist", "cifar10"}:
        return make_torchvision_image_dataset(name, n_train, n_test, noise, seed)

    builders = {
        "patch_xor": lambda count, level, current_seed: make_patch_xor_images(count, level, current_seed, image_size),
        "relative_position": lambda count, level, current_seed: make_relative_position_images(
            count, level, current_seed, image_size
        ),
    }
    if name not in builders:
        raise ValueError(f"Unknown image dataset: {name}")

    x_train, y_train = builders[name](n_train, noise, seed)
    x_test, y_test = builders[name](n_test, noise, seed + 10_000)
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


class AddGaussianNoise:
    def __init__(self, std: float, seed: int) -> None:
        self.std = std
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return image
        noisy = image + self.std * torch.randn(image.shape, generator=self.generator)
        return noisy.clamp(0.0, 1.0)


def _build_torchvision_transform(noise: float, seed: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            AddGaussianNoise(noise, seed),
        ]
    )


def _sample_subset(dataset, n_samples: int, seed: int):
    if n_samples <= 0 or n_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()
    return Subset(dataset, indices)


def make_torchvision_image_dataset(
    name: str,
    n_train: int,
    n_test: int,
    noise: float,
    seed: int,
):
    dataset_root = "data"
    dataset_builders = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
    }
    if name not in dataset_builders:
        raise ValueError(f"Unknown torchvision image dataset: {name}")

    builder = dataset_builders[name]
    train_set = builder(
        root=dataset_root,
        train=True,
        download=True,
        transform=_build_torchvision_transform(noise, seed),
    )
    test_set = builder(
        root=dataset_root,
        train=False,
        download=True,
        transform=_build_torchvision_transform(noise, seed + 10_000),
    )
    return _sample_subset(train_set, n_train, seed), _sample_subset(test_set, n_test, seed + 20_000)


class BinaryTargetDataset(Dataset):
    def __init__(self, base_dataset, positive_class: int) -> None:
        self.base_dataset = base_dataset
        self.positive_class = positive_class

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        features, label = self.base_dataset[index]
        binary_label = torch.tensor(float(label == self.positive_class), dtype=torch.float32)
        return features, binary_label


def get_image_dataset_config(name: str) -> dict[str, int]:
    configs = {
        "patch_xor": {"input_channels": 1, "num_classes": 2, "image_size": 16},
        "relative_position": {"input_channels": 1, "num_classes": 2, "image_size": 16},
        "mnist": {"input_channels": 1, "num_classes": 10, "image_size": 28},
        "cifar10": {"input_channels": 3, "num_classes": 10, "image_size": 32},
    }
    if name not in configs:
        raise ValueError(f"Unknown image dataset config: {name}")
    return configs[name]


def make_tabular_dataset(name: str, n_train: int, n_test: int, noise: float, seed: int) -> tuple[TensorDataset, TensorDataset]:
    builders = {
        "two_moons": make_two_moons,
        "checkerboard": make_checkerboard,
        "spiral": make_spiral,
    }
    if name not in builders:
        raise ValueError(f"Unknown tabular dataset: {name}")

    x_train, y_train = builders[name](n_train, noise, seed)
    x_test, y_test = builders[name](n_test, noise, seed + 10_000)
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
