# Proyecto Caso B - Predicción de compra de lujo

## 1. Objetivo
Este proyecto usa el CSV histórico de Black Friday para estimar qué clientes tienen mayor probabilidad de comprar un artículo de lujo el próximo mes.

## 2. Enfoque técnico
- **BA (Predicción):** Python con un modelo de clasificación
- **Datos:** CSV histórico en `datos/blackfriday_base.csv`

## 3. Estructura del proyecto
```text
proyecto_blackfriday/
├─ scripts/
├─ datos/
├─ modelo/
├─ README.md
└─ requirements.txt
```

## 4. Requisitos
- Python 3.10+ (recomendado)

## 5. Crear entorno virtual
### Linux o Mac
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

## 6. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 7. Flujo BA
1. Entrenar el modelo con el histórico: `python scripts/02_entrenar_modelo.py`
2. Generar el CSV final con probabilidades: `python scripts/03_generar_csv_final.py`
3. Revisar el archivo resultante en `datos/blackfriday_final_scored.csv`

## 8. Salida esperada
- Modelo entrenado:
  - `modelo/modelo_compra_lujo.joblib`
- Métricas de validación:
  - `modelo/metricas_modelo.json`
- Dataset final con score por cliente:
  - `datos/blackfriday_final_scored.csv`