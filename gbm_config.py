# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Config global consumida por Factory Maestra; sin acoplamiento a un solo backend.
# - ABSTRACCIÓN DEL DATO: Reemplazar constantes NumPy por umbrales configurables por backend cuando aplique.
# - REFACTOR NATIVO: Mantener config declarativa; backends leen límites vía inyección.
# #[AI_CONTEXT_END]
import numpy as np

from analyze_data.analyzer_factory import DataAnalyzerFactory as AnalyzerFactory

from data_cleaning.data_cleaning_pipeline import PipelineBuilder

from readers.reader_factory import ReaderFactory
"""
PIPELINE PROFESIONAL PARA ANALÍTICA DE PORTAFOLIO GBM / TRADING
===============================================================

Objetivo:
- Limpiar
- Validar
- Enriquecer
- Analizar
- Generar features financieras
- Detectar patrones
- Preparar para ML / forecasting

Compatible con:
- GBM
- Trading logs
- Portafolios de inversión
- Backtesting
"""

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = "GBM - Acciones.csv"
OUTPUT_FILE = "gbm_clean_enriched.csv"

# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # LOAD
    # =========================================================================

    print("=" * 80)
    print("READING FILE")
    print("=" * 80)

    reader = ReaderFactory.create(INPUT_FILE)
    df = reader.read()

    print(df.head())

    # =========================================================================
    # CLEANING PIPELINE
    # =========================================================================

    print("\n" + "=" * 80)
    print("BUILDING CLEANING PIPELINE")
    print("=" * 80)

    pipeline = PipelineBuilder(df).build(

        configuration=[

            # ================================================================
            # STRUCTURE
            # ================================================================

            {"fix_columns_titles": None},

            {"handle_sentinel_values": {

            }},

            # ================================================================
            # VALIDATION
            # ================================================================

            {"enforce_schema": {
                "required_columns": [
                    "fecha",
                    "ticker",
                    "tipo",
                    "moneda",
                    "titulos",
                    "monto_bruto",
                    "comision",
                    "valor_puro"
                ],
                "min_rows": 10
            }},

            # ================================================================
            # CLEANING
            # ================================================================

            {"drop_high_missing_columns": {
                "threshold": 0.60
            }},

            {"drop_constant_columns": None},

            # ================================================================
            # OUTLIERS
            # ================================================================

            {"zscore_outlier": {
                "columns": [
                    "monto_bruto",
                    "comision",
                    "valor_puro"
                ],
                "z_threshold": 3.0
            }},
        ]
    )

    # =========================================================================
    # EXECUTE PIPELINE
    # =========================================================================

    print("\nRUNNING PIPELINE...")

    result = pipeline.run(df)

    # =========================================================================
    # MANUAL FORMATTING
    # =========================================================================

    print("\nFORMATTING DATA...")

    # ---- FECHA ----

    result["fecha"] = pd.to_datetime(
        result["fecha"],
        errors="coerce",
        dayfirst=True
    )

    # ---- NUMERIC ----

    numeric_columns = [
        "titulos",
        "monto_bruto",
        "comision",
        "valor_puro",
        "precio_de_ejecucion"
    ]

    for col in numeric_columns:

        if col in result.columns:

            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            )

    # ---- DROP NULLS ----

    result = result.dropna(
        subset=[
            "fecha",
            "ticker",
            "monto_bruto"
        ]
    )

    # ---- SORT ----

    result = result.sort_values(
        by="fecha"
    ).reset_index(drop=True)

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    print("\nGENERATING FEATURES...")

    # -------------------------------------------------------------------------
    # PROFIT / COST
    # -------------------------------------------------------------------------

    result["net_amount"] = (
        result["monto_bruto"] - result["comision"]
    )

    # -------------------------------------------------------------------------
    # BUY / SELL FLAG
    # -------------------------------------------------------------------------

    result["is_buy"] = (
        result["tipo"]
        .astype(str)
        .str.lower()
        .eq("compra")
        .astype(int)
    )

    result["is_sell"] = (
        result["tipo"]
        .astype(str)
        .str.lower()
        .eq("venta")
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # FEES %
    # -------------------------------------------------------------------------

    result["commission_pct"] = (
        result["comision"] /
        result["monto_bruto"]
    ) * 100

    # -------------------------------------------------------------------------
    # DAILY OPERATIONS
    # -------------------------------------------------------------------------

    result["daily_operations"] = (
        result.groupby("fecha")["ticker"]
        .transform("count")
    )

    # -------------------------------------------------------------------------
    # TICKER FREQUENCY
    # -------------------------------------------------------------------------

    result["ticker_frequency"] = (
        result.groupby("ticker")["ticker"]
        .transform("count")
    )

    # -------------------------------------------------------------------------
    # ROLLING MEAN
    # -------------------------------------------------------------------------

    result["rolling_mean_5"] = (
        result["monto_bruto"]
        .rolling(window=5)
        .mean()
    )

    # -------------------------------------------------------------------------
    # ROLLING STD
    # -------------------------------------------------------------------------

    result["rolling_std_5"] = (
        result["monto_bruto"]
        .rolling(window=5)
        .std()
    )

    # -------------------------------------------------------------------------
    # RETURNS
    # -------------------------------------------------------------------------

    result["returns"] = (
        result["monto_bruto"]
        .pct_change()
    )

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    result["volatility_5"] = (
        result["returns"]
        .rolling(5)
        .std()
    )

    # =========================================================================
    # SAVE ENRICHED DATA
    # =========================================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nCLEAN DATA SAVED -> {OUTPUT_FILE}")

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    print("\n" + "=" * 80)
    print("STARTING ADVANCED ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # BASIC INSPECTION
    # =========================================================================

    print("\n[1] BASIC INSPECTION")
    print("-" * 50)

    print("\nDATA TYPES")
    print(
        AnalyzerFactory
        .create("types", result)
        .analyze()
    )

    print("\nSHAPE")
    print(
        AnalyzerFactory
        .create("shape", result)
        .analyze()
    )

    print("\nINFO")
    print(
        AnalyzerFactory
        .create("info", result)
        .analyze()
    )

    # =========================================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================================

    print("\n[2] DESCRIPTIVE STATISTICS")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # VALUE COUNTS
    # -------------------------------------------------------------------------

    value_counts = AnalyzerFactory.create(
        "value_counts",
        result
    )

    print("\nTOP TICKERS")
    print(
        value_counts.analyze(
            column="ticker",
            top_n=10
        )
    )

    print("\nTRANSACTION TYPES")
    print(
        value_counts.analyze(
            column="tipo"
        )
    )

    # -------------------------------------------------------------------------
    # CENTRAL TENDENCY
    # -------------------------------------------------------------------------

    central = AnalyzerFactory.create(
        "central_tendency",
        result
    )

    print("\nCENTRAL TENDENCY")
    print(
        central.analyze(
            columns=[
                "monto_bruto",
                "comision",
                "valor_puro"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # DISPERSION
    # -------------------------------------------------------------------------

    dispersion = AnalyzerFactory.create(
        "dispersion",
        result
    )

    print("\nDISPERSION")
    print(
        dispersion.analyze(
            columns=[
                "monto_bruto",
                "comision",
                "valor_puro"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # DISTRIBUTION TYPE
    # -------------------------------------------------------------------------

    distribution = AnalyzerFactory.create(
        "distribution_type",
        result
    )

    print("\nDISTRIBUTION TYPE")
    print(
        distribution.analyze(
            column="monto_bruto"
        )
    )

    # -------------------------------------------------------------------------
    # SKEWNESS / KURTOSIS
    # -------------------------------------------------------------------------

    skewness = AnalyzerFactory.create(
        "skewness_kurtosis",
        result
    )

    print("\nSKEWNESS / KURTOSIS")
    print(
        skewness.analyze(
            columns=[
                "monto_bruto",
                "comision"
            ]
        )
    )

    # =========================================================================
    # RELATIONAL ANALYSIS
    # =========================================================================

    print("\n[3] RELATIONAL ANALYSIS")
    print("-" * 50)

    correlation = AnalyzerFactory.create(
        "correlation_matrix",
        result
    )

    print("\nCORRELATION MATRIX")
    print(
        correlation.analyze(
            method="pearson"
        )
    )

    # =========================================================================
    # TIME SERIES ANALYSIS
    # =========================================================================

    print("\n[4] TIME SERIES ANALYSIS")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # TREND
    # -------------------------------------------------------------------------

    trend = AnalyzerFactory.create(
        "trend_patterns",
        result
    )

    print("\nTREND ANALYSIS")
    print(
        trend.analyze(
            x="daily_operations",
            y="monto_bruto",
            type_of_analysis="ordinary_least_squares",
            complexity="simple"
        )
    )

    # -------------------------------------------------------------------------
    # SEASONALITY
    # -------------------------------------------------------------------------

    if len(result) >= 12:

        seasonality = AnalyzerFactory.create(
            "seasonality",
            result
        )

        print("\nSEASONALITY")
        print(
            seasonality.analyze(
                target_column="monto_bruto",
                period=3
            )
        )

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    volatility = AnalyzerFactory.create(
        "volatility",
        result
    )

    print("\nVOLATILITY")
    print(
        volatility.analyze(
            date_column="fecha",
            column="monto_bruto",
            window_size=5
        )
    )

    # -------------------------------------------------------------------------
    # MOMENTUM
    # -------------------------------------------------------------------------

    momentum = AnalyzerFactory.create(
        "momentum",
        result
    )

    print("\nMOMENTUM")
    print(
        momentum.analyze(
            date_column="fecha",
            column="monto_bruto",
            period=3
        )
    )

    # -------------------------------------------------------------------------
    # MOVING AVERAGES
    # -------------------------------------------------------------------------

    moving_average = AnalyzerFactory.create(
        "moving_averages",
        result
    )

    print("\nMOVING AVERAGES")
    print(
        moving_average.analyze(
            date_column="fecha",
            column="monto_bruto",
            windows=[3, 5, 10]
        )
    )

    # =========================================================================
    # BUSINESS ANALYSIS
    # =========================================================================

    print("\n[5] BUSINESS ANALYSIS")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # PARETO
    # -------------------------------------------------------------------------

    pareto = AnalyzerFactory.create(
        "pareto_analysis",
        result
    )

    print("\nPARETO ANALYSIS")
    print(
        pareto.analyze(
            entity_column="ticker",
            value_column="monto_bruto"
        )
    )

    # -------------------------------------------------------------------------
    # RISK METRICS
    # -------------------------------------------------------------------------

    risk = AnalyzerFactory.create(
        "risk_metrics",
        result
    )

    print("\nRISK METRICS")
    print(
        risk.analyze(
            value_column="monto_bruto",
            returns_column="returns"
        )
    )

    # -------------------------------------------------------------------------
    # GROWTH RATES
    # -------------------------------------------------------------------------

    growth = AnalyzerFactory.create(
        "growth_rates",
        result
    )

    print("\nGROWTH RATES")
    print(
        growth.analyze(
            value_column="monto_bruto",
            date_column="fecha"
        )
    )

    # =========================================================================
    # ML SUPPORT
    # =========================================================================

    print("\n[6] MACHINE LEARNING SUPPORT")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # FEATURE VARIANCE
    # -------------------------------------------------------------------------

    feature_variance = AnalyzerFactory.create(
        "feature_variance",
        result
    )

    print("\nFEATURE VARIANCE")
    print(
        feature_variance.analyze()
    )

    # -------------------------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------

    dimensionality = AnalyzerFactory.create(
        "dimensionality_reduction",
        result
    )

    try:

        print("\nDIMENSIONALITY REDUCTION")
        print(
            dimensionality.analyze(
                method="pca",
                n_components=2
            )
        )

    except Exception as e:

        print(f"PCA ERROR -> {e}")

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    print("\n[7] SEGMENTATION")
    print("-" * 50)

    clustering = AnalyzerFactory.create(
        "kmeans_clusters",
        result
    )

    try:

        print("\nKMEANS CLUSTERS")
        print(
            clustering.analyze(
                columns=[
                    "monto_bruto",
                    "comision",
                    "valor_puro"
                ],
                n_clusters=3
            )
        )

    except Exception as e:

        print(f"CLUSTERING ERROR -> {e}")

    # =========================================================================
    # FINISHED
    # =========================================================================

    print("\n" + "=" * 80)
    print("FULL ANALYSIS FINISHED")
    print("=" * 80)

if __name__ == "__main__":
    main()