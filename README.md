# Proyecto Caso B - E-commerce (Black Friday + Predicción de compra de lujo)

## 1. Objetivo
Este proyecto analiza las ventas del Black Friday (BI: pasado/presente) y predice qué clientes tienen mayor probabilidad de comprar un artículo de lujo el próximo mes (BA: futuro/optimización).

## 2. Enfoque técnico
- **BI (Dashboard):** Power BI
- **BA (Predicción):** Python (modelo de clasificación)
- **Datos:** simulados y exportados a CSV

## 3. Estructura del proyecto
```text
proyecto_blackfriday/
├─ .venv/
├─ scripts/
├─ datos/
├─ modelo/
├─ dashboard/
├─ README.md
├─ .gitignore
└─ requirements.txt
```

## 4. Requisitos
- Python 3.10+ (recomendado)
- Power BI Desktop
- Git (opcional)

## 5. Crear entorno virtual
### Windows (PowerShell o CMD)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 6. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 7. Flujo de ejecución (más adelante)
1. Generar datos simulados (`scripts/01_generar_datos.py`)
2. Entrenar modelo predictivo (`scripts/02_entrenar_modelo.py`)
3. Generar CSV final con score (`scripts/03_generar_csv_final.py`)
4. Cargar `datos/blackfriday_final_scored.csv` en Power BI

## 8. KPIs BI sugeridos
- Ventas totales Black Friday
- Número de pedidos
- Ticket promedio
- Margen bruto %
- Tasa de devolución
- Clientes nuevos vs recurrentes
- Top categorías y productos
- Ventas por canal y por día/hora

## 9. Salida esperada
- Dataset final para visualización:
  - `datos/blackfriday_final_scored.csv`
- Dashboard:
  - `dashboard/blackfriday_dashboard.pbix`