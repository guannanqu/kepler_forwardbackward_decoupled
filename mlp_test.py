import argparse

import torch
from torch.utils.data import DataLoader

from datasets import make_tabular_dataset
from models import FrozenGateMLP, StandardMLP
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
        description="Compare a standard ReLU MLP against an MLP with activation regions frozen at initialization."
    )
    parser.add_argument("--dataset", choices=["two_moons", "checkerboard", "spiral"], default="spiral")
    parser.add_argument("--n-train", type=int, default=768)
    parser.add_argument("--n-test", type=int, default=1536)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument("--model-seed", type=int, default=4321)
    parser.add_argument("--wandb", action="store_true", help="Log each seed/model run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="kepler-forwardbackward-decoupled")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="mlp")
    parser.add_argument("--wandb-name-prefix", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hidden_dims = [args.width] * args.depth
    results = {"standard": [], "frozen_gate": []}
    wandb_options = WandbOptions(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name_prefix=args.wandb_name_prefix,
    )

    for seed in range(args.num_seeds):
        train_set, test_set = make_tabular_dataset(
            name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            noise=args.noise,
            seed=args.data_seed + seed,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        set_seed(args.model_seed + seed)
        standard_model = StandardMLP(2, hidden_dims, 2)
        standard_run = init_wandb_run(
            options=wandb_options,
            run_name=f"standard-seed{seed}",
            config={
                "experiment": "mlp",
                "model_type": "standard",
                "seed": seed,
                "dataset": args.dataset,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "noise": args.noise,
                "depth": args.depth,
                "width": args.width,
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
        frozen_model = FrozenGateMLP(2, hidden_dims, 2)
        frozen_run = init_wandb_run(
            options=wandb_options,
            run_name=f"frozen_gate-seed{seed}",
            config={
                "experiment": "mlp",
                "model_type": "frozen_gate",
                "seed": seed,
                "dataset": args.dataset,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "noise": args.noise,
                "depth": args.depth,
                "width": args.width,
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
            "experiment": "mlp",
            "dataset": args.dataset,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "noise": args.noise,
            "depth": args.depth,
            "width": args.width,
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
            "Fixed-region MLP: each hidden gate is computed from the initial layer weights "
            "and the current input, while the affine weights themselves still train.\n"
            f"Hidden dims: {hidden_dims}"
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        num_seeds=args.num_seeds,
    )


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
