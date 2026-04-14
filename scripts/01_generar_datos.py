import os
import numpy as np
import pandas as pd
from datetime import timedelta
import random

# --------------------------
# Configuración general
# --------------------------
np.random.seed(42)
random.seed(42)

N_CLIENTES = 2500
N_PRODUCTOS = 300
N_PEDIDOS = 15000

FECHA_INICIO = pd.Timestamp("2025-01-01")
FECHA_FIN = pd.Timestamp("2025-12-31")

# Semana Black Friday simulada
BF_INICIO = pd.Timestamp("2025-11-24")
BF_FIN = pd.Timestamp("2025-11-30")

os.makedirs("datos", exist_ok=True)

# --------------------------
# 1) Catálogo de clientes
# --------------------------
ciudades = ["CDMX", "Bogotá", "Lima", "Santiago", "Buenos Aires", "Quito", "Monterrey"]
niveles_ingreso = ["bajo", "medio", "alto"]

clientes = pd.DataFrame({
    "id_cliente": np.arange(1, N_CLIENTES + 1),
    "sexo": np.random.choice(["F", "M"], N_CLIENTES),
    "edad": np.random.randint(18, 70, N_CLIENTES),
    "ciudad": np.random.choice(ciudades, N_CLIENTES, p=[0.2,0.16,0.14,0.12,0.16,0.1,0.12]),
    "nivel_ingreso": np.random.choice(niveles_ingreso, N_CLIENTES, p=[0.35,0.45,0.20]),
    "fecha_registro": pd.to_datetime(
        np.random.randint(FECHA_INICIO.value//10**9, FECHA_FIN.value//10**9, N_CLIENTES),
        unit="s"
    )
})

# --------------------------
# 2) Catálogo de productos
# --------------------------
categorias = ["Electrónica", "Moda", "Hogar", "Belleza", "Deportes", "Accesorios"]
marcas = ["Nova", "UrbanX", "Elite", "Prime", "Luxor", "BasicCo"]

es_lujo = np.random.choice([0,1], N_PRODUCTOS, p=[0.8,0.2])

precios = []
costos = []
for l in es_lujo:
    if l == 1:
        p = np.round(np.random.uniform(200, 1400), 2)
    else:
        p = np.round(np.random.uniform(10, 280), 2)
    c = np.round(p * np.random.uniform(0.45, 0.75), 2)
    precios.append(p)
    costos.append(c)

productos = pd.DataFrame({
    "id_producto": np.arange(1, N_PRODUCTOS + 1),
    "categoria": np.random.choice(categorias, N_PRODUCTOS),
    "marca": np.random.choice(marcas, N_PRODUCTOS),
    "es_lujo": es_lujo,
    "precio_base": precios,
    "costo_unitario": costos
})

# --------------------------
# Función fecha con pico BF
# --------------------------
def fecha_con_pico_blackfriday():
    # 20% de pedidos en semana BF para generar pico
    if np.random.rand() < 0.20:
        dias = (BF_FIN - BF_INICIO).days
        return BF_INICIO + pd.Timedelta(
            days=np.random.randint(0, dias + 1),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
    else:
        dias = (FECHA_FIN - FECHA_INICIO).days
        return FECHA_INICIO + pd.Timedelta(
            days=np.random.randint(0, dias + 1),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )

# --------------------------
# 3) Generar pedidos + líneas
# (1 fila = 1 ítem de pedido)
# --------------------------
canales = ["web", "app", "tienda"]
fuentes = ["orgánico", "ads", "email", "social", "referido"]
dispositivos = ["mobile", "desktop", "tablet"]

rows = []
id_pedido = 1

for _ in range(N_PEDIDOS):
    id_cliente = np.random.randint(1, N_CLIENTES + 1)
    fecha_pedido = fecha_con_pico_blackfriday()
    canal = np.random.choice(canales, p=[0.58, 0.32, 0.10])
    fuente = np.random.choice(fuentes, p=[0.30, 0.25, 0.15, 0.20, 0.10])
    dispositivo = np.random.choice(dispositivos, p=[0.65, 0.30, 0.05])

    # estado pedido
    estado = np.random.choice(["entregado", "cancelado"], p=[0.94, 0.06])

    # descuento mayor en BF
    if BF_INICIO <= fecha_pedido.normalize() <= BF_FIN:
        descuento = np.round(np.random.uniform(0.10, 0.45), 2)
    else:
        descuento = np.round(np.random.uniform(0.00, 0.25), 2)

    # ítems por pedido
    n_items = np.random.randint(1, 5)
    prods = np.random.choice(productos["id_producto"], n_items, replace=False)

    for p in prods:
        prod = productos.loc[productos["id_producto"] == p].iloc[0]
        cantidad = np.random.randint(1, 3)

        # precio final con variación
        precio_unit = np.round(prod["precio_base"] * np.random.uniform(0.90, 1.05), 2)
        monto_bruto = np.round(cantidad * precio_unit, 2)
        monto_neto = np.round(monto_bruto * (1 - descuento), 2)
        costo_linea = np.round(cantidad * prod["costo_unitario"], 2)
        margen_linea = np.round(monto_neto - costo_linea, 2)

        # devoluciones aprox 8% (solo entregados)
        if estado == "entregado" and np.random.rand() < 0.08:
            fue_devuelto = 1
            monto_devolucion = np.round(monto_neto * np.random.uniform(0.2, 1.0), 2)
        else:
            fue_devuelto = 0
            monto_devolucion = 0.0

        # score simulado inicial (luego se reemplaza con modelo real)
        base_score = (
            0.15
            + (0.20 if prod["es_lujo"] == 1 else 0.0)
            + (0.10 if clientes.loc[clientes["id_cliente"] == id_cliente, "nivel_ingreso"].values[0] == "alto" else 0.0)
            + np.random.uniform(0, 0.35)
        )
        score_lujo = float(np.clip(base_score, 0, 0.99))

        if score_lujo >= 0.75:
            segmento = "alto"
            accion = "Oferta premium personalizada"
        elif score_lujo >= 0.50:
            segmento = "medio"
            accion = "Retargeting + email de lujo"
        else:
            segmento = "bajo"
            accion = "Campaña de nutrición"

        row = {
            "id_pedido": id_pedido,
            "fecha_pedido": fecha_pedido,
            "id_cliente": id_cliente,
            "sexo": clientes.loc[clientes["id_cliente"] == id_cliente, "sexo"].values[0],
            "edad": int(clientes.loc[clientes["id_cliente"] == id_cliente, "edad"].values[0]),
            "ciudad": clientes.loc[clientes["id_cliente"] == id_cliente, "ciudad"].values[0],
            "nivel_ingreso": clientes.loc[clientes["id_cliente"] == id_cliente, "nivel_ingreso"].values[0],
            "canal": canal,
            "fuente_trafico": fuente,
            "dispositivo": dispositivo,
            "estado_pedido": estado,
            "descuento": descuento,
            "id_producto": int(prod["id_producto"]),
            "categoria": prod["categoria"],
            "marca": prod["marca"],
            "es_lujo": int(prod["es_lujo"]),
            "cantidad": int(cantidad),
            "precio_unitario": float(precio_unit),
            "costo_unitario": float(prod["costo_unitario"]),
            "monto_linea_bruto": float(monto_bruto),
            "monto_linea_neto": float(monto_neto),
            "costo_linea": float(costo_linea),
            "margen_linea": float(margen_linea),
            "fue_devuelto": int(fue_devuelto),
            "monto_devolucion_linea": float(monto_devolucion),
            "score_lujo": round(score_lujo, 4),
            "segmento_score": segmento,
            "accion_recomendada": accion
        }
        rows.append(row)

    id_pedido += 1

df = pd.DataFrame(rows)

# Guardar CSV base
ruta_salida = "./datos/pedidos_simulados.csv" 
df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

print(f"✅ Archivo generado: {ruta_salida}")
print(f"Filas: {len(df):,}")
print("Columnas:", list(df.columns))