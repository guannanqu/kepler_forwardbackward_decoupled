import argparse

import cnn_test
import mlp_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run either the MLP or CNN frozen-gate experiment."
    )
    parser.add_argument("experiment", choices=["mlp", "cnn"], help="Which experiment to run.")
    args, remaining = parser.parse_known_args()

    if args.experiment == "mlp":
        mlp_test.run_experiment(mlp_test.build_parser().parse_args(remaining))
        return

    cnn_test.run_experiment(cnn_test.build_parser().parse_args(remaining))


if __name__ == "__main__":
    main()
