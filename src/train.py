from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import DiabetesMLP


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    learning_rate: float
    weight_decay: float
    epochs: int = 100
    batch_size: int = 32
    hidden1: int = 32
    hidden2: int = 16
    seed: int = 42


def set_seed(seed: int) -> None:
    """Fixa as sementes do NumPy e PyTorch para tornar as execuções reproduzíveis.
    Não reinicia `random.seed` global do Python pois o OpenTelemetry/MLflow usa
    `random.getrandbits` para gerar `trace_id` e `span_id` únicos por run.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _to_tensor(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, criterion: nn.Module) -> dict:
    """Avalia a MLP sem atualizar pesos e retorna métricas e predições."""
    model.eval()
    X_t, y_t = _to_tensor(X, y)
    with torch.no_grad():
        logits = model(X_t)
        loss = criterion(logits, y_t).item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int().cpu().numpy()

    y_int = y.astype(int)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_int, preds)),
        "precision": float(precision_score(y_int, preds, zero_division=0)),
        "recall": float(recall_score(y_int, preds, zero_division=0)),
        "f1": float(f1_score(y_int, preds, zero_division=0)),
        "preds": preds,
        "probs": probs.cpu().numpy(),
    }


def _save_loss_plot(history: dict, run_artifacts: Path, run_name: str) -> Path:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Curvas de perda - {run_name}")
    plt.legend()
    plt.tight_layout()
    path = run_artifacts / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_validation_metrics_plot(history: dict, run_artifacts: Path, run_name: str) -> Path:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.plot(history["val_f1"], label="val_f1")
    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.ylim(0.0, 1.0)
    plt.title(f"Métricas de validação - {run_name}")
    plt.legend()
    plt.tight_layout()
    path = run_artifacts / "validation_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_one_run(config: ExperimentConfig, data, artifact_root: str | Path) -> dict:
    """Treina uma configuração, registra métricas no MLflow e salva evidências."""
    set_seed(config.seed)
    run_artifacts = Path(artifact_root) / config.name
    run_artifacts.mkdir(parents=True, exist_ok=True)

    model = DiabetesMLP(
        input_dim=data.X_train.shape[1],
        hidden1=config.hidden1,
        hidden2=config.hidden2,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_train_t, y_train_t = _to_tensor(data.X_train, data.y_train)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    # Um único span de treinamento deixa o trace legível; as métricas continuam por época.
    with mlflow.start_span(name="train") as train_span:
        train_span.set_inputs(
            {
                "run_name": config.name,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
            }
        )
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(xb)

            train_loss = running_loss / len(loader.dataset)
            val = evaluate(model, data.X_val, data.y_val, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
            history["val_accuracy"].append(val["accuracy"])
            history["val_f1"].append(val["f1"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "val_f1": val["f1"],
                },
                step=epoch,
            )
        train_span.set_outputs(
            {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_val_accuracy": history["val_accuracy"][-1],
                "final_val_f1": history["val_f1"][-1],
            }
        )

    # O span de validação representa a avaliação final usada para comparar as runs.
    with mlflow.start_span(name="validate") as val_span:
        val_span.set_inputs({"run_name": config.name, "n_val": int(len(data.y_val))})
        final_val = evaluate(model, data.X_val, data.y_val, criterion)
        val_span.set_outputs(
            {
                "val_loss": final_val["loss"],
                "val_accuracy": final_val["accuracy"],
                "val_precision": final_val["precision"],
                "val_recall": final_val["recall"],
                "val_f1": final_val["f1"],
            }
        )

    mlflow.log_metrics(
        {
            "final_val_loss": final_val["loss"],
            "final_val_accuracy": final_val["accuracy"],
            "final_val_precision": final_val["precision"],
            "final_val_recall": final_val["recall"],
            "final_val_f1": final_val["f1"],
        }
    )

    loss_plot = _save_loss_plot(history, run_artifacts, config.name)
    metrics_plot = _save_validation_metrics_plot(history, run_artifacts, config.name)

    model_path = run_artifacts / "model_state_dict.pt"
    torch.save(model.state_dict(), model_path)

    # Cada run possui seu próprio modelo e suas próprias curvas como artefatos.
    mlflow.log_artifact(str(loss_plot), artifact_path="plots")
    mlflow.log_artifact(str(metrics_plot), artifact_path="plots")
    mlflow.log_artifact(str(model_path), artifact_path="model")

    return {
        "model": model,
        "model_path": model_path,
        "loss_plot": loss_plot,
        "metrics_plot": metrics_plot,
        "history": history,
        "final_val": final_val,
    }
