from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Espelho público amplamente usado do Pima Indians Diabetes Dataset.
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    imputer: SimpleImputer
    split_summary: dict


def ensure_dataset(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset não encontrado em {path}. Baixando de {DATA_URL} ...")
    try:
        urlretrieve(DATA_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Não foi possível baixar o dataset automaticamente. "
            f"Baixe o Pima Indians Diabetes Dataset e salve em: {path}"
        ) from exc
    return path


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    path = ensure_dataset(csv_path)
    df = pd.read_csv(path)

    # O espelho acima não possui cabeçalho. Se o CSV local já possuir, preservamos.
    if "Outcome" not in df.columns:
        df = pd.read_csv(path, header=None, names=COLUMNS)

    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    return df[COLUMNS].copy()


def prepare_data(csv_path: str | Path, seed: int = 42) -> PreparedData:
    df = load_dataframe(csv_path)

    X = df.drop(columns="Outcome").copy()
    y = df["Outcome"].astype(int).to_numpy()

    # Zeros clinicamente inválidos são tratados como ausentes nessas medições.
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    # 70% treino, 15% validação, 15% teste, mantendo proporção da classe.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    # IMPORTANTÍSSIMO: imputer e scaler são ajustados SOMENTE no treino.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

    summary = {
        "seed": seed,
        "n_total": int(len(df)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "test_positive_rate": float(np.mean(y_test)),
        "zero_as_missing": ZERO_AS_MISSING,
        "imputation": "median fitted on train only",
        "scaling": "StandardScaler fitted on train only",
        "split": "70/15/15 stratified",
    }

    return PreparedData(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        y_test=y_test.astype(np.float32),
        scaler=scaler,
        imputer=imputer,
        split_summary=summary,
    )


def save_preprocessors(data: PreparedData, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaler_path = out / "scaler.joblib"
    imputer_path = out / "imputer.joblib"
    joblib.dump(data.scaler, scaler_path)
    joblib.dump(data.imputer, imputer_path)
    return scaler_path, imputer_path
