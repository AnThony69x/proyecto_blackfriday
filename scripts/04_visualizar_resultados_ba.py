from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datos" / "blackfriday_final_scored.csv"
OUTPUT_DIR = PROJECT_ROOT / "resultados_ba"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {DATA_PATH}. Ejecuta primero scripts/03_generar_csv_final.py"
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"id_cliente", "probabilidad_compra_lujo", "gasto_total_neto"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing)}")

    return df.sort_values("probabilidad_compra_lujo", ascending=False).reset_index(drop=True)


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.family": "DejaVu Sans",
        }
    )


def draw_distribution_chart(ax: plt.Axes, df: pd.DataFrame) -> None:
    base_blue = "#0B1F3A"
    dark_gray = "#4B5563"
    bright_gold = "#D4A72C"
    soft_gold = "#F4D58D"

    sns.histplot(
        data=df,
        x="probabilidad_compra_lujo",
        bins=24,
        stat="count",
        kde=True,
        color=base_blue,
        edgecolor="white",
        alpha=0.88,
        ax=ax,
    )

    kde_line = ax.lines[0]
    kde_x = np.asarray(kde_line.get_xdata())
    kde_y = np.asarray(kde_line.get_ydata())
    opportunity_threshold = 0.60
    vip_threshold = 0.80

    lower_mask = kde_x <= opportunity_threshold
    upper_mask = kde_x >= opportunity_threshold

    ax.fill_between(kde_x[lower_mask], 0, kde_y[lower_mask], color=dark_gray, alpha=0.28, zorder=1)
    ax.fill_between(kde_x[upper_mask], 0, kde_y[upper_mask], color=bright_gold, alpha=0.35, zorder=1)

    ax.axvline(opportunity_threshold, color=soft_gold, linestyle="--", linewidth=2.5)
    ax.axvline(vip_threshold, color=bright_gold, linestyle="--", linewidth=2.5)

    y_max = ax.get_ylim()[1]
    ax.text(
        opportunity_threshold + 0.01,
        y_max * 0.80,
        "Target de Oportunidad\n(Top 10%)",
        ha="left",
        va="top",
        color=soft_gold,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=soft_gold, alpha=0.95),
    )
    ax.text(
        vip_threshold + 0.005,
        y_max * 0.93,
        "Target VIP\n80%",
        ha="left",
        va="top",
        color=bright_gold,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=bright_gold, alpha=0.95),
    )
    ax.text(
        0.03,
        y_max * 0.18,
        "Nota: El umbral VIP del 80% está vacío.\nSe recomienda ajustar la campaña a clientes con score > 0.6\npara capturar volumen de ventas.",
        ha="left",
        va="center",
        color="#111827",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CBD5E1", alpha=0.98),
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad de compra de lujo")
    ax.set_ylabel("Número de clientes")
    ax.set_title("Oportunidad de Negocio: Ajustando el umbral de campaña")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_distribution_chart(ax, df)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_top_clients_chart(ax: plt.Axes, df: pd.DataFrame, title: str, label_size: int) -> None:
    base_blue = "#0B1F3A"
    bright_gold = "#D4A72C"
    soft_gray = "#94A3B8"
    blue_potential = "#3B82F6"

    data = df.copy()
    probability_threshold = 0.60
    spend_threshold = float(data["gasto_total_neto"].quantile(0.60))

    data["etiqueta_color"] = data.apply(
        lambda row: bright_gold if row["probabilidad_compra_lujo"] > probability_threshold and row["gasto_total_neto"] >= spend_threshold else (
            blue_potential if row["probabilidad_compra_lujo"] > probability_threshold else soft_gray
        ),
        axis=1,
    )

    def point_color(row: pd.Series) -> str:
        if row["probabilidad_compra_lujo"] > probability_threshold and row["gasto_total_neto"] >= spend_threshold:
            return bright_gold
        if row["probabilidad_compra_lujo"] > probability_threshold and row["gasto_total_neto"] < spend_threshold:
            return blue_potential
        return soft_gray

    bubble_colors = data.apply(point_color, axis=1)
    spend_min = float(data["gasto_total_neto"].min())
    spend_max = float(data["gasto_total_neto"].max())
    size_min, size_max = 120, 1200
    bubble_sizes = np.interp(data["gasto_total_neto"], (spend_min, spend_max), (size_min, size_max))

    ax.scatter(
        data["gasto_total_neto"],
        data["probabilidad_compra_lujo"],
        s=bubble_sizes,
        c=bubble_colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    top_10 = data.nlargest(10, "probabilidad_compra_lujo").copy()
    for _, row in top_10.iterrows():
        x_offset = (spend_max - spend_min) * 0.012
        y_offset = 0.004
        ax.text(
            row["gasto_total_neto"] + x_offset,
            row["probabilidad_compra_lujo"] + y_offset,
            f"ID {int(row['id_cliente'])}",
            fontsize=label_size,
            fontweight="bold",
            color="#111827",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#E2E8F0", alpha=0.95),
            zorder=4,
        )

    ax.scatter([], [], s=260, c=bright_gold, edgecolors="white", linewidth=1.0, label="VIP: alto gasto y alta prob.")
    ax.scatter([], [], s=260, c=blue_potential, edgecolors="white", linewidth=1.0, label="Alta prob. / gasto bajo")
    ax.scatter([], [], s=260, c=soft_gray, edgecolors="white", linewidth=1.0, label="Resto")

    ax.annotate(
        "Clientes VIP: Máxima Prioridad",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        xytext=(0.74, 0.88),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=bright_gold, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=bright_gold, alpha=0.98),
        fontsize=12,
        fontweight="bold",
        color="#7A4E00",
    )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_title(title)
    ax.set_xlabel("Gasto Histórico del Black Friday")
    ax.set_ylabel("Probabilidad de Compra de Lujo")
    ax.set_xlim(spend_min * 0.95, spend_max * 1.05)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Señal de Valor",
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#CBD5E1",
        fontsize=11,
        title_fontsize=12,
        markerscale=1.3,
        borderpad=0.8,
    )


def plot_top_clients(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_top_clients_chart(
        ax,
        df,
        "Mapa de Calor: ¿A quién llamar mañana mismo?",
        label_size=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_scatter_action_chart(ax: plt.Axes, df: pd.DataFrame, title: str, vip_label_size: int) -> None:
    base_blue = "#0B1F3A"
    soft_blue = "#DCEEFF"
    soft_gold = "#F8E7B5"
    accent_gold = "#D4A72C"

    data = df.copy()
    score_col = "probabilidad_compra_lujo"
    spend_col = "gasto_total_neto"

    score_p50 = float(data[score_col].quantile(0.5))
    spend_p50 = float(data[spend_col].quantile(0.5))
    spend_mean = float(data[spend_col].mean())

    x_min, x_max = float(data[spend_col].min()), float(data[spend_col].max())
    y_min, y_max = float(data[score_col].min()), float(data[score_col].max())

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.08
    x_min_plot = max(0.0, x_min - x_pad)
    x_max_plot = x_max + x_pad
    y_min_plot = max(0.0, y_min - y_pad)
    y_max_plot = min(1.0, y_max + y_pad)

    # Cuadrantes destacados: VIP (arriba-derecha) y Recuperacion (abajo-derecha).
    ax.axvspan(spend_p50, x_max_plot, ymin=0, ymax=1, facecolor="#FFFFFF", alpha=0.0, zorder=0)
    ax.fill_between(
        [spend_p50, x_max_plot],
        score_p50,
        y_max_plot,
        color=soft_gold,
        alpha=0.30,
        zorder=0,
    )
    ax.fill_between(
        [spend_p50, x_max_plot],
        y_min_plot,
        score_p50,
        color=soft_blue,
        alpha=0.35,
        zorder=0,
    )

    def action_label(row: pd.Series) -> str:
        if row[score_col] >= score_p50 and row[spend_col] >= spend_p50:
            return "Campaña Exclusiva"
        if row[score_col] < score_p50 and row[spend_col] >= spend_p50:
            return "Cupón de Reactivación"
        if row[score_col] >= score_p50 and row[spend_col] < spend_p50:
            return "Upselling Selectivo"
        return "Nutrición Automatizada"

    data["accion_segmento"] = data.apply(action_label, axis=1)
    action_palette = {
        "Campaña Exclusiva": accent_gold,
        "Cupón de Reactivación": "#2563EB",
        "Upselling Selectivo": "#0EA5A4",
        "Nutrición Automatizada": base_blue,
    }

    sns.scatterplot(
        data=data,
        x=spend_col,
        y=score_col,
        hue="accion_segmento",
        palette=action_palette,
        alpha=0.80,
        s=75,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
        zorder=2,
    )

    ax.axvline(spend_p50, color="#94A3B8", linestyle="--", linewidth=1.3)
    ax.axhline(score_p50, color="#94A3B8", linestyle="--", linewidth=1.3)
    ax.axvline(spend_mean, color="#64748B", linestyle=":", linewidth=1.6)

    vip_points = data[(data[score_col] >= score_p50) & (data[spend_col] >= spend_p50)]
    vip_top = vip_points.nlargest(min(8, len(vip_points)), score_col)
    for _, row in vip_top.iterrows():
        ax.annotate(
            f"ID {int(row['id_cliente'])}",
            (row[spend_col], row[score_col]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=vip_label_size,
            color="#7A4E00",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#E5C76B", alpha=0.95),
        )

    ax.text(
        spend_p50 + (x_max_plot - spend_p50) * 0.03,
        score_p50 + (y_max_plot - score_p50) * 0.90,
        "Campeones (VIP)",
        color="#7A4E00",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        spend_p50 + (x_max_plot - spend_p50) * 0.03,
        y_min_plot + (score_p50 - y_min_plot) * 0.10,
        "Gigantes Dormidos (Recuperación)",
        color="#1E40AF",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        spend_mean,
        y_min_plot + (y_max_plot - y_min_plot) * 0.03,
        "Gasto promedio",
        color="#334155",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#94A3B8", alpha=0.95),
    )

    ax.set_title(title)
    ax.set_xlabel("Gasto histórico neto")
    ax.set_ylabel("Probabilidad de compra de lujo")
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Acción Recomendada",
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#CBD5E1",
        fontsize=13,
        title_fontsize=14,
        markerscale=1.8,
        borderpad=1,
    )


def plot_gasto_vs_probabilidad(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_scatter_action_chart(
        ax,
        df,
        "Relación entre gasto histórico y probabilidad de compra",
        vip_label_size=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_executive_panel(df: pd.DataFrame, output_path: Path) -> None:
    base_blue = "#0B1F3A"
    accent_gold = "#D4A72C"
    soft_gold = "#F8E7B5"
    soft_blue = "#DCEEFF"

    fig = plt.figure(figsize=(20, 17), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.14, 0.42, 0.44], hspace=0.32, wspace=0.18)

    ax_banner = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    score_col = "probabilidad_compra_lujo"
    spend_col = "gasto_total_neto"
    score_p50 = float(df[score_col].quantile(0.5))
    opportunity_clients = int((df[score_col] > 0.6).sum())
    dormant_mask = (df[score_col] < score_p50) & (df[spend_col] >= float(df[spend_col].quantile(0.5)))
    recovered_spend = float(df.loc[dormant_mask, spend_col].sum())
    vip_mask = (df[score_col] >= score_p50) & (df[spend_col] >= float(df[spend_col].quantile(0.5)))
    vip_ticket_avg = float(df.loc[vip_mask, spend_col].mean()) if vip_mask.any() else 0.0

    ax_banner.axis("off")
    banner_cards = [
        (0.02, 0.18, 0.29, 0.64, "Volumen de Oportunidad", f"{opportunity_clients:,}", base_blue),
        (0.355, 0.18, 0.29, 0.64, "Gasto Potencial Recuperable", f"${recovered_spend:,.0f}", soft_blue),
        (0.69, 0.18, 0.29, 0.64, "Ticket Promedio VIP", f"${vip_ticket_avg:,.0f}", accent_gold),
    ]
    for x, y, w, h, label, value, color in banner_cards:
        ax_banner.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                transform=ax_banner.transAxes,
                facecolor="white",
                edgecolor="#CBD5E1",
                linewidth=1.4,
                zorder=1,
            )
        )
        ax_banner.add_patch(
            plt.Rectangle(
                (x, y + h - 0.10),
                w,
                0.10,
                transform=ax_banner.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=2,
            )
        )
        ax_banner.text(
            x + 0.03,
            y + 0.42,
            label,
            transform=ax_banner.transAxes,
            fontsize=13,
            fontweight="bold",
            color="#111827",
            va="center",
        )
        ax_banner.text(
            x + 0.03,
            y + 0.18,
            value,
            transform=ax_banner.transAxes,
            fontsize=22,
            fontweight="bold",
            color=color,
            va="center",
        )
    ax_banner.text(
        0.02,
        1.04,
        "Reporte Ejecutivo de Decisiones | Black Friday BA",
        transform=ax_banner.transAxes,
        fontsize=20,
        fontweight="bold",
        color=base_blue,
        va="bottom",
    )

    draw_distribution_chart(ax1, df)
    ax1.annotate(
        "ZONA CRÍTICA: Ajustar pauta aquí",
        xy=(0.60, ax1.get_ylim()[1] * 0.55),
        xytext=(0.14, ax1.get_ylim()[1] * 0.80),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=accent_gold, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor=accent_gold, alpha=0.97),
        fontsize=12,
        fontweight="bold",
        color="#7A4E00",
    )

    draw_top_clients_chart(
        ax2,
        df,
        "Mapa de Calor: ¿A quién llamar mañana mismo?",
        label_size=10,
    )

    draw_scatter_action_chart(
        ax3,
        df,
        "Gasto histórico vs probabilidad",
        vip_label_size=10,
    )
    ax3.annotate(
        "PRIORIDAD: 340 clientes a reactivar",
        xy=(float(df[spend_col].quantile(0.75)), score_p50 * 0.82),
        xytext=(float(df[spend_col].quantile(0.82)), score_p50 * 0.70),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=base_blue, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#CBD5E1", alpha=0.97),
        fontsize=12,
        fontweight="bold",
        color=base_blue,
    )

    fig.suptitle(
        "Black Friday BA | Clientes con mayor probabilidad de compra de lujo",
        fontsize=23,
        fontweight="bold",
        color=base_blue,
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        "Estrategia Black Friday: Priorizar volumen en score > 0.6 y reactivación de alto gasto.",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#334155",
    )
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.98)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    df = load_data()

    plot_distribution(df, OUTPUT_DIR / "01_distribucion_score.png")
    plot_top_clients(df, OUTPUT_DIR / "02_top_10_clientes.png")
    plot_gasto_vs_probabilidad(df, OUTPUT_DIR / "03_gasto_vs_probabilidad.png")
    build_executive_panel(df, OUTPUT_DIR / "panel_ejecutivo_ba.png")

    print(f"Gráficos generados en: {OUTPUT_DIR}")
    print("Archivos:")
    print(f"- {OUTPUT_DIR / '01_distribucion_score.png'}")
    print(f"- {OUTPUT_DIR / '02_top_10_clientes.png'}")
    print(f"- {OUTPUT_DIR / '03_gasto_vs_probabilidad.png'}")
    print(f"- {OUTPUT_DIR / 'panel_ejecutivo_ba.png'}")


if __name__ == "__main__":
    main()
