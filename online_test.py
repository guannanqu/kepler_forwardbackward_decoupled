import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import BinaryTargetDataset, get_image_dataset_config, make_image_dataset
from models import FrozenGateCNN, StandardCNN
from trainers import WandbOptions, format_mean_std, init_wandb_run, log_wandb_history_summary, set_seed


@dataclass
class OnlineEpochMetrics:
    global_epoch: int
    task_index: int
    positive_class: int
    train_loss: float
    train_acc: float
    current_task_acc: float
    avg_seen_tasks_acc: float


@dataclass
class OnlineRunMetrics:
    mean_current_task_acc: float
    final_avg_seen_tasks_acc: float
    task_rows: list[dict[str, float | int]]
    history: list[OnlineEpochMetrics]


def parse_task_order(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def evaluate_binary_task(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features).squeeze(-1)
            predictions = (logits >= 0).to(labels.dtype)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.numel()

    return total_correct / total_examples


def evaluate_binary_loss_and_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            predictions = (logits >= 0).to(labels.dtype)

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_binary_task(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return optimizer


def build_model(args: argparse.Namespace, dataset_config: dict[str, int], model_type: str) -> nn.Module:
    model_cls = StandardCNN if model_type == "standard" else FrozenGateCNN
    return model_cls(
        image_size=args.image_size or dataset_config["image_size"],
        num_classes=1,
        channels=list(args.channels),
        input_channels=dataset_config["input_channels"],
        pooling=args.pooling,
        use_residual=args.residual,
        convs_per_stage=args.convs_per_stage,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continual binary-task experiment for comparing standard and frozen-gate CNNs."
    )
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--task-order", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-tasks", type=int, default=5, help="Use the first K classes from --task-order.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--channels", type=int, nargs=2, default=[12, 24])
    parser.add_argument("--convs-per-stage", type=int, default=2)
    parser.add_argument("--pooling", choices=["none", "avg", "max"], default="max")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--epochs-per-task", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=3141)
    parser.add_argument("--model-seed", type=int, default=2718)
    parser.add_argument("--wandb", action="store_true", help="Log each model/seed online-learning run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="frozen_gage_NN_plasticity")
    parser.add_argument("--wandb-entity", default="gqu-carnegie-mellon-university")
    parser.add_argument("--wandb-group", default="online")
    parser.add_argument("--wandb-name-prefix", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_config = get_image_dataset_config(args.dataset)
    task_order = parse_task_order(args.task_order)[: args.num_tasks]
    wandb_options = WandbOptions(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name_prefix=args.wandb_name_prefix,
    )

    results = {
        "standard": [],
        "frozen_gate": [],
    }

    for seed in range(args.num_seeds):
        train_base, test_base = make_image_dataset(
            name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            noise=args.noise,
            seed=args.data_seed + seed,
            image_size=args.image_size or dataset_config["image_size"],
        )

        for model_type in ["standard", "frozen_gate"]:
            set_seed(args.model_seed + seed)
            model = build_model(args, dataset_config, model_type).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()
            wandb_run = init_wandb_run(
                options=wandb_options,
                run_name=f"{model_type}-seed{seed}",
                config={
                    "experiment": "online",
                    "model_type": model_type,
                    "seed": seed,
                    "dataset": args.dataset,
                    "task_order": task_order,
                    "num_tasks": len(task_order),
                    "epochs_per_task": args.epochs_per_task,
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "noise": args.noise,
                    "channels": list(args.channels),
                    "convs_per_stage": args.convs_per_stage,
                    "pooling": args.pooling,
                    "residual": args.residual,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                },
            )

            per_task_rows: list[dict[str, float | int]] = []
            current_task_accs: list[float] = []
            epoch_history: list[OnlineEpochMetrics] = []
            global_epoch = 0

            for task_index, positive_class in enumerate(task_order):
                train_dataset = BinaryTargetDataset(train_base, positive_class)
                test_datasets = {
                    seen_class: BinaryTargetDataset(test_base, seen_class)
                    for seen_class in task_order[: task_index + 1]
                }
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loaders = {
                    seen_class: DataLoader(task_dataset, batch_size=args.batch_size, shuffle=False)
                    for seen_class, task_dataset in test_datasets.items()
                }

                for _ in range(args.epochs_per_task):
                    model.train()
                    for features, labels in train_loader:
                        features = features.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        logits = model(features).squeeze(-1)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()

                    train_loss, train_acc = evaluate_binary_loss_and_accuracy(model, train_loader, device)
                    seen_task_accs = {
                        seen_class: evaluate_binary_task(model, loader, device)
                        for seen_class, loader in test_loaders.items()
                    }
                    current_task_acc = seen_task_accs[positive_class]
                    avg_seen_tasks_acc = sum(seen_task_accs.values()) / len(seen_task_accs)
                    global_epoch += 1
                    epoch_history.append(
                        OnlineEpochMetrics(
                            global_epoch=global_epoch,
                            task_index=task_index + 1,
                            positive_class=positive_class,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            current_task_acc=current_task_acc,
                            avg_seen_tasks_acc=avg_seen_tasks_acc,
                        )
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "global_epoch": global_epoch,
                                "task_index": task_index + 1,
                                "positive_class": positive_class,
                                "train_loss": train_loss,
                                "train_acc": train_acc,
                                "current_task_acc": current_task_acc,
                                "avg_seen_tasks_acc": avg_seen_tasks_acc,
                            }
                        )

                seen_task_accs = {
                    seen_class: evaluate_binary_task(model, loader, device)
                    for seen_class, loader in test_loaders.items()
                }
                current_task_acc = seen_task_accs[positive_class]
                avg_seen_tasks_acc = sum(seen_task_accs.values()) / len(seen_task_accs)
                current_task_accs.append(current_task_acc)
                per_task_rows.append(
                    {
                        "task_index": task_index + 1,
                        "positive_class": positive_class,
                        "current_task_acc": current_task_acc,
                        "avg_seen_tasks_acc": avg_seen_tasks_acc,
                    }
                )

            run_metrics = OnlineRunMetrics(
                mean_current_task_acc=sum(current_task_accs) / len(current_task_accs),
                final_avg_seen_tasks_acc=per_task_rows[-1]["avg_seen_tasks_acc"],
                task_rows=per_task_rows,
                history=epoch_history,
            )
            results[model_type].append(run_metrics)
            if wandb_run is not None:
                wandb_run.summary["mean_current_task_acc"] = run_metrics.mean_current_task_acc
                wandb_run.summary["final_avg_seen_tasks_acc"] = run_metrics.final_avg_seen_tasks_acc
                wandb_run.finish()

    if args.wandb:
        history_results = {
            model_type: [
                type("HistoryWrapper", (), {
                    "history": [
                        type(
                            "HistoryEpoch",
                            (),
                            {
                                "train_loss": epoch.train_loss,
                                "train_acc": epoch.train_acc,
                                "valid_acc": epoch.avg_seen_tasks_acc,
                            },
                        )()
                        for epoch in model_runs.history
                    ]
                })()
                for model_runs in results[model_type]
            ]
            for model_type in ["standard", "frozen_gate"]
        }
        log_wandb_history_summary(
            options=wandb_options,
            run_name="online-history-summary",
            config={
                "experiment": "online",
                "dataset": args.dataset,
                "task_order": task_order,
                "num_tasks": len(task_order),
                "epochs_per_task": args.epochs_per_task,
                "summary_type": "mean_std_across_seeds",
            },
            results=history_results,
        )

    print(f"Dataset: {args.dataset}")
    print(
        f"Task order: {task_order}, epochs/task: {args.epochs_per_task}, channels: {list(args.channels)}, "
        f"convs/stage: {args.convs_per_stage}, pooling: {args.pooling}, residual: {args.residual}"
    )
    print("")
    print(f"{'model':<14}{'mean current acc':<24}{'final avg seen acc':<24}")
    print("-" * 62)
    for model_type in ["standard", "frozen_gate"]:
        mean_current = [run.mean_current_task_acc for run in results[model_type]]
        final_seen = [run.final_avg_seen_tasks_acc for run in results[model_type]]
        print(
            f"{model_type:<14}"
            f"{format_mean_std(mean_current):<24}"
            f"{format_mean_std(final_seen):<24}"
        )

    print("")
    print("Per-task mean accuracy across seeds:")
    for model_type in ["standard", "frozen_gate"]:
        print(f"{model_type}:")
        for task_index, positive_class in enumerate(task_order):
            current_values = [
                run.task_rows[task_index]["current_task_acc"]
                for run in results[model_type]
            ]
            seen_values = [
                run.task_rows[task_index]["avg_seen_tasks_acc"]
                for run in results[model_type]
            ]
            print(
                f"  task {task_index + 1} (class {positive_class}): "
                f"current={format_mean_std(current_values)}, "
                f"avg_seen={format_mean_std(seen_values)}"
            )


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
