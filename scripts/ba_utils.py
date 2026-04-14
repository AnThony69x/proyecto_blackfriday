from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datos" / "blackfriday_base.csv"


def load_transactions(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carga el CSV base y normaliza tipos clave."""

    transactions = pd.read_csv(path)
    transactions["fecha_pedido"] = pd.to_datetime(transactions["fecha_pedido"])
    transactions = transactions.sort_values(
        ["fecha_pedido", "id_cliente", "id_pedido"]
    ).reset_index(drop=True)
    return transactions


def _mode_value(values: pd.Series):
    cleaned = values.dropna()
    if cleaned.empty:
        return np.nan

    mode = cleaned.mode()
    if mode.empty:
        return cleaned.iloc[0]
    return mode.iloc[0]


def _mean_gap_days(values: pd.Series) -> float:
    ordered = pd.Series(values).sort_values()
    if ordered.size < 2:
        return 0.0

    gaps = ordered.diff().dt.days.dropna()
    if gaps.empty:
        return 0.0
    return float(gaps.mean())


def build_customer_snapshot_features(
    transactions: pd.DataFrame,
    snapshot_end: pd.Timestamp,
) -> pd.DataFrame:
    """Construye variables por cliente usando solo el historial hasta un corte."""

    snapshot_end = pd.Timestamp(snapshot_end)

    history = transactions[transactions["fecha_pedido"] <= snapshot_end].copy()
    if history.empty:
        return pd.DataFrame()

    history["monto_lujo_neto"] = history["monto_linea_neto"] * history["es_lujo"]

    order_summary = (
        history.groupby(["id_cliente", "id_pedido"], as_index=False)
        .agg(
            fecha_pedido=("fecha_pedido", "max"),
            monto_pedido_neto=("monto_linea_neto", "sum"),
            lujo_pedido=("es_lujo", "max"),
        )
        .sort_values(["id_cliente", "fecha_pedido"])
    )

    customer_base = history.groupby("id_cliente", sort=False).agg(
        sexo=("sexo", _mode_value),
        edad=("edad", "median"),
        ciudad=("ciudad", _mode_value),
        nivel_ingreso=("nivel_ingreso", _mode_value),
        canal_principal=("canal", _mode_value),
        fuente_principal=("fuente_trafico", _mode_value),
        dispositivo_principal=("dispositivo", _mode_value),
        categoria_principal=("categoria", _mode_value),
        marca_principal=("marca", _mode_value),
        total_lineas=("id_pedido", "size"),
        gasto_total_neto=("monto_linea_neto", "sum"),
        gasto_promedio_linea=("monto_linea_neto", "mean"),
        margen_total=("margen_linea", "sum"),
        margen_promedio=("margen_linea", "mean"),
        descuento_promedio=("descuento", "mean"),
        tasa_devolucion=("fue_devuelto", "mean"),
        lineas_devueltas=("fue_devuelto", "sum"),
        luxury_lines=("es_lujo", "sum"),
        luxury_share_linea=("es_lujo", "mean"),
        gasto_lujo_neto=("monto_lujo_neto", "sum"),
        monto_devolucion_total=("monto_devolucion_linea", "sum"),
        compras_con_descuento=("descuento", lambda values: float((values > 0).mean())),
        meses_activos=("fecha_pedido", lambda values: values.dt.to_period("M").nunique()),
    )

    customer_orders = order_summary.groupby("id_cliente", sort=False).agg(
        total_pedidos=("id_pedido", "nunique"),
        ticket_medio_pedido=("monto_pedido_neto", "mean"),
        ticket_max_pedido=("monto_pedido_neto", "max"),
        luxury_orders=("lujo_pedido", "sum"),
        luxury_share_pedido=("lujo_pedido", "mean"),
        primera_compra=("fecha_pedido", "min"),
        ultima_compra=("fecha_pedido", "max"),
        dias_promedio_entre_compras=("fecha_pedido", _mean_gap_days),
    )

    features = customer_base.join(customer_orders, how="left")
    features["snapshot_end"] = snapshot_end
    features["antiguedad_cliente_dias"] = (
        snapshot_end - features["primera_compra"]
    ).dt.days
    features["recencia_dias"] = (snapshot_end - features["ultima_compra"]).dt.days
    features["pedidos_por_mes"] = features["total_pedidos"] / features["meses_activos"].clip(lower=1)
    features["gasto_por_mes"] = features["gasto_total_neto"] / features["meses_activos"].clip(lower=1)
    features["gasto_por_pedido"] = features["gasto_total_neto"] / features["total_pedidos"].clip(lower=1)
    features["luxury_order_share"] = features["luxury_orders"] / features["total_pedidos"].clip(lower=1)
    features["gasto_lujo_share"] = features["gasto_lujo_neto"] / features["gasto_total_neto"].replace(0, np.nan)

    features = features.drop(columns=["primera_compra", "ultima_compra"])
    features = features.fillna(
        {
            "gasto_lujo_share": 0.0,
            "dias_promedio_entre_compras": 0.0,
        }
    )
    return features.reset_index()


def build_labeled_dataset(transactions: pd.DataFrame) -> pd.DataFrame:
    """Construye un dataset por cliente y mes con etiqueta de compra de lujo al mes siguiente."""

    month_ends = [pd.Timestamp(value) for value in sorted(
        transactions["fecha_pedido"].dt.to_period("M").dt.to_timestamp("M").unique()
    )]
    if len(month_ends) < 3:
        raise ValueError("Se necesitan al menos 3 meses de datos para entrenar el modelo.")

    samples: list[pd.DataFrame] = []
    for snapshot_end, target_month_end in zip(month_ends[:-1], month_ends[1:]):
        features = build_customer_snapshot_features(transactions, snapshot_end)
        if features.empty:
            continue

        future = transactions[
            (transactions["fecha_pedido"] > snapshot_end)
            & (transactions["fecha_pedido"] <= target_month_end)
        ]
        target = (
            future.groupby("id_cliente")["es_lujo"]
            .max()
            .rename("target_compra_lujo")
            .reset_index()
        )

        sample = features.merge(target, on="id_cliente", how="left")
        sample["target_compra_lujo"] = sample["target_compra_lujo"].fillna(0).astype(int)
        sample["snapshot_end"] = snapshot_end
        sample["target_month_end"] = target_month_end
        samples.append(sample)

    if not samples:
        raise ValueError("No se pudieron construir muestras de entrenamiento.")

    return pd.concat(samples, ignore_index=True)
