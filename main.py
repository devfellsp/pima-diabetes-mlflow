import argparse

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline observável - Pima Indians Diabetes + PyTorch + MLflow"
    )
    parser.add_argument("--data", default="data/diabetes.csv", help="Caminho do CSV")
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="URI do backend do MLflow",
    )
    args = parser.parse_args()

    run_pipeline(csv_path=args.data, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
