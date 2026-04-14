from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	average_precision_score,
	classification_report,
	roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from ba_utils import PROJECT_ROOT, build_labeled_dataset, load_transactions


MODEL_DIR = PROJECT_ROOT / "modelo"
MODEL_PATH = MODEL_DIR / "modelo_compra_lujo.joblib"
METRICS_PATH = MODEL_DIR / "metricas_modelo.json"
TARGET_COLUMN = "target_compra_lujo"
IGNORE_COLUMNS = {TARGET_COLUMN, "id_cliente", "snapshot_end", "target_month_end"}


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"encoder",
				OneHotEncoder(handle_unknown="ignore", sparse_output=True),
			),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_features),
			("cat", categorical_pipeline, categorical_features),
		]
	)

	model = LogisticRegression(
		class_weight="balanced",
		max_iter=2000,
		solver="liblinear",
	)

	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", model),
		]
	)


def main() -> None:
	transactions = load_transactions()
	dataset = build_labeled_dataset(transactions)

	snapshot_months = sorted(dataset["snapshot_end"].unique())
	if len(snapshot_months) < 2:
		raise ValueError("No hay suficientes meses para separar entrenamiento y validación.")

	validation_snapshot = pd.Timestamp(snapshot_months[-1])
	train_dataset = dataset[dataset["snapshot_end"] < validation_snapshot].copy()
	validation_dataset = dataset[dataset["snapshot_end"] == validation_snapshot].copy()

	feature_frame = dataset.drop(columns=list(IGNORE_COLUMNS), errors="ignore")
	numeric_features = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
	categorical_features = [column for column in feature_frame.columns if column not in numeric_features]

	X_train = train_dataset.drop(columns=list(IGNORE_COLUMNS), errors="ignore")
	y_train = train_dataset[TARGET_COLUMN]
	X_validation = validation_dataset.drop(columns=list(IGNORE_COLUMNS), errors="ignore")
	y_validation = validation_dataset[TARGET_COLUMN]

	pipeline = build_pipeline(numeric_features, categorical_features)
	pipeline.fit(X_train, y_train)

	validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
	inverted_probabilities = 1.0 - validation_probabilities
	raw_auc = roc_auc_score(y_validation, validation_probabilities)
	inverted_auc = roc_auc_score(y_validation, inverted_probabilities)
	score_transform = "invert" if inverted_auc > raw_auc else "direct"
	selected_probabilities = (
		inverted_probabilities if score_transform == "invert" else validation_probabilities
	)
	validation_predictions = (selected_probabilities >= 0.5).astype(int)

	metrics = {
		"train_rows": int(len(train_dataset)),
		"validation_rows": int(len(validation_dataset)),
		"validation_snapshot": str(validation_snapshot.date()),
		"train_positive_rate": float(y_train.mean()),
		"validation_positive_rate": float(y_validation.mean()),
		"roc_auc_raw": float(raw_auc),
		"roc_auc_adjusted": float(roc_auc_score(y_validation, selected_probabilities)),
		"average_precision_adjusted": float(average_precision_score(y_validation, selected_probabilities)),
		"score_transform": score_transform,
		"classification_report": classification_report(
			y_validation,
			validation_predictions,
			output_dict=True,
			zero_division=0,
		),
	}

	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(
		{
			"pipeline": pipeline,
			"numeric_features": numeric_features,
			"categorical_features": categorical_features,
			"validation_snapshot": str(validation_snapshot.date()),
			"score_transform": score_transform,
		},
		MODEL_PATH,
	)

	with METRICS_PATH.open("w", encoding="utf-8") as file_handle:
		json.dump(metrics, file_handle, ensure_ascii=False, indent=2)

	print(f"Modelo guardado en: {MODEL_PATH}")
	print(f"Métricas guardadas en: {METRICS_PATH}")
	print(f"ROC AUC crudo: {metrics['roc_auc_raw']:.4f}")
	print(f"ROC AUC ajustado: {metrics['roc_auc_adjusted']:.4f}")
	print(f"Average precision ajustado: {metrics['average_precision_adjusted']:.4f}")
	print(f"Transformación de score: {score_transform}")


if __name__ == "__main__":
	main()

