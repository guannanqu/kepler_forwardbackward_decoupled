import argparse

import torch
from torch.utils.data import DataLoader

from datasets import get_image_dataset_config, make_image_dataset
from models import FrozenGateCNN, StandardCNN
from trainers import (
    WandbOptions,
    finish_wandb_run,
    init_wandb_run,
    log_wandb_history_summary,
    print_summary,
    set_seed,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a standard CNN against a CNN with ReLU gates and max-pool routing frozen at initialization."
    )
    parser.add_argument("--dataset", choices=["patch_xor", "relative_position", "mnist", "cifar10"], default="relative_position")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=0, help="0 means use the full training split.")
    parser.add_argument("--n-test", type=int, default=0, help="0 means use the full test split.")
    parser.add_argument("--noise", type=float, default=0.28, help="Gaussian pixel noise. Also applies to MNIST/CIFAR-10.")
    parser.add_argument("--channels", type=int, nargs=2, default=[12, 24])
    parser.add_argument("--convs-per-stage", type=int, default=1, help="Number of conv+ReLU layers in each CNN stage.")
    parser.add_argument("--pooling", choices=["none", "avg", "max"], default="max")
    parser.add_argument("--residual", action="store_true", help="Enable residual shortcuts around the two conv blocks.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=2468)
    parser.add_argument("--model-seed", type=int, default=8642)
    parser.add_argument("--wandb", action="store_true", help="Log each seed/model run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="frozen_gate_NN")
    parser.add_argument("--wandb-entity", default="gqu-carnegie-mellon-university")
    parser.add_argument("--wandb-group", default="cnn")
    parser.add_argument("--wandb-name-prefix", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    channels = list(args.channels)
    dataset_config = get_image_dataset_config(args.dataset)
    image_size = args.image_size or dataset_config["image_size"]
    input_channels = dataset_config["input_channels"]
    num_classes = dataset_config["num_classes"]
    results = {"standard": [], "frozen_gate": []}
    wandb_options = WandbOptions(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_name_prefix,
        name_prefix=args.wandb_name_prefix,
    )

    for seed in range(args.num_seeds):
        train_set, test_set = make_image_dataset(
            name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            noise=args.noise,
            seed=args.data_seed + seed,
            image_size=image_size,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        set_seed(args.model_seed + seed)
        standard_model = StandardCNN(
            image_size,
            num_classes,
            channels,
            input_channels=input_channels,
            pooling=args.pooling,
            use_residual=args.residual,
            convs_per_stage=args.convs_per_stage,
        )
        standard_run = init_wandb_run(
            options=wandb_options,
            run_name=f"standard-seed{seed}",
            config={
                "experiment": "cnn",
                "model_type": "standard",
                "seed": seed,
                "dataset": args.dataset,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "noise": args.noise,
                "channels": channels,
                "convs_per_stage": args.convs_per_stage,
                "pooling": args.pooling,
                "residual": args.residual,
                "image_size": image_size,
                "input_channels": input_channels,
                "num_classes": num_classes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
            },
        )
        standard_metrics = train_model(
            model=standard_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            wandb_run=standard_run,
        )
        finish_wandb_run(
            standard_run,
            {
                "final_train_loss": standard_metrics.final_train_loss,
                "final_train_acc": standard_metrics.final_train_acc,
                "final_test_acc": standard_metrics.final_test_acc,
            },
        )
        results["standard"].append(standard_metrics)

        set_seed(args.model_seed + seed)
        frozen_model = FrozenGateCNN(
            image_size,
            num_classes,
            channels,
            input_channels=input_channels,
            pooling=args.pooling,
            use_residual=args.residual,
            convs_per_stage=args.convs_per_stage,
        )
        frozen_run = init_wandb_run(
            options=wandb_options,
            run_name=f"frozen_gate-seed{seed}",
            config={
                "experiment": "cnn",
                "model_type": "frozen_gate",
                "seed": seed,
                "dataset": args.dataset,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "noise": args.noise,
                "channels": channels,
                "convs_per_stage": args.convs_per_stage,
                "pooling": args.pooling,
                "residual": args.residual,
                "image_size": image_size,
                "input_channels": input_channels,
                "num_classes": num_classes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
            },
        )
        frozen_metrics = train_model(
            model=frozen_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            wandb_run=frozen_run,
        )
        finish_wandb_run(
            frozen_run,
            {
                "final_train_loss": frozen_metrics.final_train_loss,
                "final_train_acc": frozen_metrics.final_train_acc,
                "final_test_acc": frozen_metrics.final_test_acc,
            },
        )
        results["frozen_gate"].append(frozen_metrics)

    log_wandb_history_summary(
        options=wandb_options,
        run_name="history-summary",
        config={
            "experiment": "cnn",
            "dataset": args.dataset,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "noise": args.noise,
            "channels": channels,
            "convs_per_stage": args.convs_per_stage,
            "pooling": args.pooling,
            "residual": args.residual,
            "image_size": image_size,
            "input_channels": input_channels,
            "num_classes": num_classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_seeds": args.num_seeds,
            "summary_type": "mean_std_across_seeds",
        },
        results=results,
    )

    print_summary(
        title=f"Dataset: {args.dataset}",
        results=results,
        device=device,
        model_details=(
            "Frozen-gate CNN: ReLU masks and 2x2 max-pool winner indices are both chosen "
            "from the initialization-time activations, while the trainable convolutions and classifier update normally.\n"
            f"Channels: {channels}, image size: {image_size}, input channels: {input_channels}, "
            f"classes: {num_classes}, convs/stage: {args.convs_per_stage}, pooling: {args.pooling}, residual: {args.residual}"
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        num_seeds=args.num_seeds,
    )


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
