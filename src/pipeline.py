from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn

from .data import prepare_data, save_preprocessors
from .train import ExperimentConfig, evaluate, train_one_run


EXPERIMENTS = [
    ExperimentConfig(name="baseline", learning_rate=0.01, weight_decay=0.0),
    ExperimentConfig(name="lr_menor", learning_rate=0.001, weight_decay=0.0),
    ExperimentConfig(name="regularizacao_l2", learning_rate=0.01, weight_decay=1e-5),
]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline(
    csv_path: str = "data/diabetes.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "Pima_Indians_Diabetes_MLP",
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)

    results: list[dict] = []

    for cfg in EXPERIMENTS:
        with mlflow.start_run(run_name=cfg.name) as run:
            mlflow.set_tags(
                {
                    "dataset": "Pima Indians Diabetes",
                    "model_type": "PyTorch MLP",
                    "task": "binary_classification",
                    "selection_rule": "validation_f1_then_accuracy",
                }
            )
            mlflow.log_params(
                {
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "architecture": f"8-{cfg.hidden1}-{cfg.hidden2}-1",
                    "seed": cfg.seed,
                    "split": "70/15/15 stratified",
                    "optimizer": "Adam",
                    "loss": "BCEWithLogitsLoss",
                }
            )

            # O preparo fica dentro da própria run para que o trace tenha
            # prepare_data -> train -> validate de forma clara e reproduzível.
            # A mesma seed garante o mesmo split nas três configurações.
            with mlflow.start_span(name="prepare_data") as prep_span:
                prep_span.set_inputs({"csv_path": str(csv_path), "seed": cfg.seed})
                data = prepare_data(csv_path, seed=cfg.seed)

                run_artifacts = artifact_root / cfg.name
                run_artifacts.mkdir(parents=True, exist_ok=True)

                preprocessing_summary = run_artifacts / "preprocessing_summary.json"
                _write_json(preprocessing_summary, data.split_summary)
                scaler_path, imputer_path = save_preprocessors(data, run_artifacts)
                prep_span.set_outputs(data.split_summary)

            mlflow.log_artifact(str(preprocessing_summary), artifact_path="preprocessing")
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
            mlflow.log_artifact(str(imputer_path), artifact_path="preprocessing")

            trained = train_one_run(cfg, data, artifact_root)
            results.append(
                {
                    "config": cfg,
                    "run_id": run.info.run_id,
                    "data": data,
                    **trained,
                }
            )

    # Escolha SEM olhar o teste: F1 de validação, com acurácia como desempate.
    best = max(
        results,
        key=lambda r: (r["final_val"]["f1"], r["final_val"]["accuracy"]),
    )

    # O conjunto de teste da configuração escolhida é usado UMA ÚNICA VEZ.
    best_data = best["data"]
    with mlflow.start_run(run_id=best["run_id"]):
        with mlflow.start_span(name="final_test") as test_span:
            test_span.set_inputs(
                {
                    "selected_run": best["config"].name,
                    "n_test": int(len(best_data.y_test)),
                }
            )
            test = evaluate(
                best["model"],
                best_data.X_test,
                best_data.y_test,
                nn.BCEWithLogitsLoss(),
            )
            test_span.set_outputs(
                {
                    "test_loss": test["loss"],
                    "test_accuracy": test["accuracy"],
                    "test_precision": test["precision"],
                    "test_recall": test["recall"],
                    "test_f1": test["f1"],
                }
            )

        mlflow.set_tag("selected_model", "true")
        mlflow.log_metrics(
            {
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
            }
        )

        # Evidências finais apenas para o modelo selecionado.
        final_dir = artifact_root / "final_test"
        final_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ConfusionMatrixDisplay.from_predictions(
            best_data.y_test.astype(int),
            test["preds"],
            ax=ax,
            values_format="d",
        )
        plt.title(f"Matriz de confusão - {best['config'].name}")
        plt.tight_layout()
        cm_path = final_dir / "confusion_matrix_test.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        report = classification_report(
            best_data.y_test.astype(int),
            test["preds"],
            output_dict=True,
            zero_division=0,
        )
        report_path = final_dir / "classification_report_test.json"
        _write_json(report_path, report)

        selected_model_path = final_dir / "selected_model_state_dict.pt"
        torch.save(best["model"].state_dict(), selected_model_path)

        mlflow.log_artifact(str(cm_path), artifact_path="final_test")
        mlflow.log_artifact(str(report_path), artifact_path="final_test")
        mlflow.log_artifact(str(selected_model_path), artifact_path="selected_model")

    comparison = {
        r["config"].name: {
            "run_id": r["run_id"],
            "val_loss": r["final_val"]["loss"],
            "val_accuracy": r["final_val"]["accuracy"],
            "val_precision": r["final_val"]["precision"],
            "val_recall": r["final_val"]["recall"],
            "val_f1": r["final_val"]["f1"],
        }
        for r in results
    }
    summary = {
        "selected_run": best["config"].name,
        "selected_run_id": best["run_id"],
        "selection_basis": "highest validation F1; accuracy as tiebreaker",
        "comparison": comparison,
        "final_test": {
            "loss": test["loss"],
            "accuracy": test["accuracy"],
            "precision": test["precision"],
            "recall": test["recall"],
            "f1": test["f1"],
        },
    }
    summary_path = artifact_root / "experiment_summary.json"
    _write_json(summary_path, summary)

    # Também registra o resumo comparativo na run vencedora.
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_artifact(str(summary_path), artifact_path="summary")

    print("\n=== RESULTADO ===")
    for name, metrics in comparison.items():
        print(
            f"{name:18s} | val_acc={metrics['val_accuracy']:.4f} "
            f"| val_precision={metrics['val_precision']:.4f} "
            f"| val_recall={metrics['val_recall']:.4f} "
            f"| val_f1={metrics['val_f1']:.4f}"
        )
    print(f"\nModelo escolhido: {summary['selected_run']}")
    print(
        f"Teste final: accuracy={test['accuracy']:.4f} "
        f"| precision={test['precision']:.4f} "
        f"| recall={test['recall']:.4f} "
        f"| f1={test['f1']:.4f}"
    )
    print("Abra o MLflow com: mlflow ui --backend-store-uri sqlite:///mlflow.db")

    return summary
