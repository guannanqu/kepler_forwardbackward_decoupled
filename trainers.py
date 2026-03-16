import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    valid_acc: float


@dataclass
class RunMetrics:
    final_train_loss: float
    final_train_acc: float
    final_test_acc: float
    history: list[EpochMetrics]


@dataclass
class WandbOptions:
    enabled: bool
    project: str
    entity: str | None
    group: str | None
    name_prefix: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def init_wandb_run(
    *,
    options: WandbOptions | None,
    run_name: str,
    config: dict[str, object],
):
    if options is None or not options.enabled:
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging was requested, but the `wandb` package is not installed. "
            "Install it with `pip install wandb` and try again."
        ) from error

    name = run_name if options.name_prefix is None else f"{options.name_prefix}-{run_name}"
    return wandb.init(
        project=options.project,
        entity=options.entity,
        group=options.group,
        name=name,
        config=config,
        reinit=True,
    )


def finish_wandb_run(run, metrics: dict[str, float]) -> None:
    if run is None:
        return
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.finish()


def log_wandb_history_summary(
    *,
    options: WandbOptions | None,
    run_name: str,
    config: dict[str, object],
    results: dict[str, list[RunMetrics]],
) -> None:
    if options is None or not options.enabled:
        return

    run = init_wandb_run(options=options, run_name=run_name, config=config)
    if run is None:
        return

    metric_names = ["train_loss", "train_acc", "valid_acc"]
    table_columns = ["epoch", "model_type", "metric", "mean", "std", "lower", "upper"]
    table_rows: list[list[object]] = []

    num_epochs = len(next(iter(results.values()))[0].history)
    for epoch_index in range(num_epochs):
        log_payload: dict[str, float | int] = {"epoch": epoch_index + 1}
        for model_name, model_runs in results.items():
            for metric_name in metric_names:
                values = [
                    getattr(model_run.history[epoch_index], metric_name)
                    for model_run in model_runs
                ]
                values_tensor = torch.tensor(values, dtype=torch.float32)
                mean = values_tensor.mean().item()
                std = values_tensor.std(unbiased=False).item()
                lower = mean - std
                upper = mean + std

                prefix = f"{model_name}/{metric_name}"
                log_payload[f"{prefix}_mean"] = mean
                log_payload[f"{prefix}_std"] = std
                log_payload[f"{prefix}_lower"] = lower
                log_payload[f"{prefix}_upper"] = upper
                table_rows.append([epoch_index + 1, model_name, metric_name, mean, std, lower, upper])

        run.log(log_payload)

    try:
        import wandb

        summary_table = wandb.Table(columns=table_columns, data=table_rows)
        run.log({"history_summary_table": summary_table})
    except ImportError:
        pass

    run.finish()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    wandb_run=None,
) -> RunMetrics:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    history: list[EpochMetrics] = []

    for epoch_id in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        _, test_acc = evaluate(model, test_loader, device)
        history.append(EpochMetrics(epoch=epoch_id + 1, train_loss=train_loss, train_acc=train_acc, valid_acc=test_acc))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch_id + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_acc": test_acc,
                }
            )

    final_metrics = history[-1]
    print(
        f"Epoch {final_metrics.epoch}/{epochs}: "
        f"train loss={final_metrics.train_loss:.4f}, "
        f"train acc={final_metrics.train_acc:.4f}, "
        f"test acc={final_metrics.valid_acc:.4f}"
    )
    return RunMetrics(
        final_train_loss=final_metrics.train_loss,
        final_train_acc=final_metrics.train_acc,
        final_test_acc=final_metrics.valid_acc,
        history=history,
    )


def format_mean_std(values: list[float]) -> str:
    values_tensor = torch.tensor(values, dtype=torch.float32)
    mean = values_tensor.mean().item()
    std = values_tensor.std(unbiased=False).item()
    return f"{mean:.4f} +/- {std:.4f}"


def print_summary(
    *,
    title: str,
    results: dict[str, list[RunMetrics]],
    device: torch.device,
    model_details: str,
    epochs: int,
    learning_rate: float,
    num_seeds: int,
) -> None:
    print(title)
    print(model_details)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, lr: {learning_rate}, seeds: {num_seeds}")
    print("")
    print(f"{'model':<14}{'train loss':<22}{'train acc':<22}{'test acc':<22}")
    print("-" * 80)

    for model_name in ["standard", "frozen_gate"]:
        metrics = results[model_name]
        train_losses = [entry.final_train_loss for entry in metrics]
        train_accs = [entry.final_train_acc for entry in metrics]
        test_accs = [entry.final_test_acc for entry in metrics]
        print(
            f"{model_name:<14}"
            f"{format_mean_std(train_losses):<22}"
            f"{format_mean_std(train_accs):<22}"
            f"{format_mean_std(test_accs):<22}"
        )
