import copy

import torch
import torch.nn.functional as F
from torch import nn


class StandardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[index], dims[index + 1]) for index in range(len(dims) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class FrozenGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[index], dims[index + 1]) for index in range(len(dims) - 1))
        self.initial_layers = copy.deepcopy(self.layers)

        for layer in self.initial_layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for current_layer, initial_layer in zip(self.layers[:-1], self.initial_layers[:-1]):
            current_preact = current_layer(x)
            with torch.no_grad():
                initial_preact = initial_layer(x)
                gate = (initial_preact > 0).to(current_preact.dtype)
            x = current_preact * gate
        return self.layers[-1](x)


def frozen_max_pool2d(
    current: torch.Tensor,
    reference: torch.Tensor,
    kernel_size: int = 2,
    stride: int = 2,
) -> torch.Tensor:
    current_windows = F.unfold(current, kernel_size=kernel_size, stride=stride)
    reference_windows = F.unfold(reference, kernel_size=kernel_size, stride=stride)

    batch_size, channels_times_kernel, n_windows = current_windows.shape
    channels = current.shape[1]
    kernel_elements = kernel_size * kernel_size

    current_windows = current_windows.view(batch_size, channels, kernel_elements, n_windows)
    reference_windows = reference_windows.view(batch_size, channels, kernel_elements, n_windows)
    indices = reference_windows.argmax(dim=2, keepdim=True)
    pooled = current_windows.gather(dim=2, index=indices).squeeze(2)

    out_height = (current.shape[2] - kernel_size) // stride + 1
    out_width = (current.shape[3] - kernel_size) // stride + 1
    return pooled.view(batch_size, channels, out_height, out_width)


def pooled_spatial_size(image_size: int, pooling: str, num_pool_layers: int = 2) -> int:
    if pooling == "none":
        return image_size
    return image_size // (2**num_pool_layers)


def apply_pool(x: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "none":
        return x
    if pooling == "avg":
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    if pooling == "max":
        return F.max_pool2d(x, kernel_size=2, stride=2)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def build_stage_layers(input_channels: int, output_channels: int, convs_per_stage: int) -> nn.ModuleList:
    layers = [nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)]
    for _ in range(convs_per_stage - 1):
        layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1))
    return nn.ModuleList(layers)


class StandardCNN(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        channels: list[int],
        input_channels: int = 1,
        pooling: str = "max",
        use_residual: bool = False,
        convs_per_stage: int = 1,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_residual = use_residual
        self.stage1 = build_stage_layers(input_channels, channels[0], convs_per_stage)
        self.stage2 = build_stage_layers(channels[0], channels[1], convs_per_stage)
        self.shortcut1 = nn.Conv2d(input_channels, channels[0], kernel_size=1)
        self.shortcut2 = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        pooled_size = pooled_spatial_size(image_size, pooling)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels[1] * pooled_size * pooled_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = apply_pool(self.shortcut1(x), self.pooling)
        for layer in self.stage1:
            x = F.relu(layer(x))
        x = apply_pool(x, self.pooling)
        if self.use_residual:
            x = x + residual

        residual = apply_pool(self.shortcut2(x), self.pooling)
        for layer in self.stage2:
            x = F.relu(layer(x))
        x = apply_pool(x, self.pooling)
        if self.use_residual:
            x = x + residual

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FrozenGateCNN(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        channels: list[int],
        input_channels: int = 1,
        pooling: str = "max",
        use_residual: bool = False,
        convs_per_stage: int = 1,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_residual = use_residual
        self.stage1 = build_stage_layers(input_channels, channels[0], convs_per_stage)
        self.stage2 = build_stage_layers(channels[0], channels[1], convs_per_stage)
        self.shortcut1 = nn.Conv2d(input_channels, channels[0], kernel_size=1)
        self.shortcut2 = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        pooled_size = pooled_spatial_size(image_size, pooling)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels[1] * pooled_size * pooled_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.initial_stage1 = copy.deepcopy(self.stage1)
        self.initial_stage2 = copy.deepcopy(self.stage2)
        self.initial_fc1 = copy.deepcopy(self.fc1)
        for layer_group in [self.initial_stage1, self.initial_stage2, [self.initial_fc1]]:
            for layer in layer_group:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def _forward_frozen_stage(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        current_layers: nn.ModuleList,
        reference_layers: nn.ModuleList,
        shortcut: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = apply_pool(shortcut(current), self.pooling)
        for current_layer, reference_layer in zip(current_layers, reference_layers):
            current = current_layer(current)
            with torch.no_grad():
                reference = reference_layer(reference)
                relu_gate = (reference > 0).to(current.dtype)
            current = current * relu_gate
            reference = reference * relu_gate

        if self.pooling == "max":
            current = frozen_max_pool2d(current, reference, kernel_size=2, stride=2)
            reference = F.max_pool2d(reference, kernel_size=2, stride=2)
        else:
            current = apply_pool(current, self.pooling)
            reference = apply_pool(reference, self.pooling)
        if self.use_residual:
            current = current + residual
        return current, reference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current, reference = self._forward_frozen_stage(
            x,
            x,
            self.stage1,
            self.initial_stage1,
            self.shortcut1,
        )
        current, reference = self._forward_frozen_stage(
            current,
            reference,
            self.stage2,
            self.initial_stage2,
            self.shortcut2,
        )

        current = self.flatten(current)
        reference = self.flatten(reference)
        current_hidden = self.fc1(current)
        with torch.no_grad():
            reference_hidden = self.initial_fc1(reference)
            relu_gate = (reference_hidden > 0).to(current_hidden.dtype)
        current_hidden = current_hidden * relu_gate
        return self.fc2(current_hidden)
