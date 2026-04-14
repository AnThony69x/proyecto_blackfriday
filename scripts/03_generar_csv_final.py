from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from ba_utils import PROJECT_ROOT, build_customer_snapshot_features, load_transactions


MODEL_PATH = PROJECT_ROOT / "modelo" / "modelo_compra_lujo.joblib"
OUTPUT_PATH = PROJECT_ROOT / "datos" / "blackfriday_final_scored.csv"


def build_segment(probability: float) -> tuple[str, str]:
	if probability >= 0.75:
		return "alto", "Oferta VIP y contacto personalizado"
	if probability >= 0.45:
		return "medio", "Retargeting con catálogo premium"
	return "bajo", "Campaña de nutrición y fidelización"


def main() -> None:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"No se encontró el modelo entrenado en {MODEL_PATH}. Ejecuta primero scripts/02_entrenar_modelo.py"
		)

	artifact = joblib.load(MODEL_PATH)
	pipeline = artifact["pipeline"]
	score_transform = artifact.get("score_transform", "direct")

	transactions = load_transactions()
	month_ends = [pd.Timestamp(value) for value in sorted(
		transactions["fecha_pedido"].dt.to_period("M").dt.to_timestamp("M").unique()
	)]
	if len(month_ends) < 2:
		raise ValueError("No hay suficientes meses para generar la predicción final.")

	scoring_snapshot = month_ends[-2]
	feature_frame = build_customer_snapshot_features(transactions, scoring_snapshot)
	if feature_frame.empty:
		raise ValueError("No se pudieron construir clientes para el scoring final.")

	X_score = feature_frame.drop(columns=["id_cliente", "snapshot_end"], errors="ignore")
	probabilities = pipeline.predict_proba(X_score)[:, 1]
	if score_transform == "invert":
		probabilities = 1.0 - probabilities

	scored = feature_frame.copy()
	scored["probabilidad_compra_lujo"] = probabilities
	scored[["segmento_score", "accion_recomendada"]] = scored["probabilidad_compra_lujo"].apply(
		lambda value: pd.Series(build_segment(float(value)))
	)

	output_columns = [
		"id_cliente",
		"probabilidad_compra_lujo",
		"segmento_score",
		"accion_recomendada",
		"sexo",
		"edad",
		"ciudad",
		"nivel_ingreso",
		"total_pedidos",
		"gasto_total_neto",
		"gasto_lujo_neto",
		"gasto_lujo_share",
		"luxury_order_share",
		"recencia_dias",
		"antiguedad_cliente_dias",
		"pedidos_por_mes",
		"ticket_medio_pedido",
	]

	scored = scored[output_columns].sort_values("probabilidad_compra_lujo", ascending=False)
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	scored.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

	print(f"CSV final generado en: {OUTPUT_PATH}")
	print("Top 10 clientes con mayor probabilidad:")
	print(scored[["id_cliente", "probabilidad_compra_lujo", "segmento_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
	main()

