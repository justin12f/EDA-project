"""Main module for running the EDA project"""

from analyze_data.data_analyzer_factory import AnalyzerFactory
from data_cleaning.data_cleaning_pipeline import PipelineBuilder
from parsers.parser import parser
from readers.reader_factory import ReaderFactory


def main():
    """Main function for running the EDA project"""
    args = parser("shopping_trends.csv")

    if args.output is None:
        args.output = f"clean_{args.input}"
    if args.report is None:
        args.report = f"cleaning_report_{args.input}.json"

    print(f"Reading data from {args.input}...")
    reader = ReaderFactory.create(args.input)
    data = reader.read()
    print(data)
    print(f"Building pipeline using preset: '{args.preset}'...")
    pipeline = PipelineBuilder(data).build(
    configuration=[

        # ── 1. ESTRUCTURA ───────────────────────────────
        {"fix_columns_titles": None},
        {"handle_sentinel_values": None},

        # ── 2. VALIDACIÓN BASE ──────────────────────────
        {"enforce_schema": {
            "required_columns": [
                "customer_id", "age", "gender",
                "purchase_amount_(usd)", "category", "review_rating"
            ],
            "min_rows": 1,
        }},

        # ── 3. LIMPIEZA BÁSICA ──────────────────────────
        {"drop_high_missing_columns": {"threshold": 0.85}},
        {"drop_constant_columns": None},

        # ── 4. BOOLEANOS ────────────────────────────────
        {
            "fix_bools_columns": {
                "columns": [
                    "discount_applied",
                    "promo_code_used",
                    "subscription_status"
                ]
            }
        },

        # ── 5. CONVERSIÓN GENERAL ───────────────────────
        {
            "safe_conversion": {
                "columns": [
                    "age",
                    "purchase_amount_(usd)",
                    "review_rating",
                    "previous_purchases"
                ]
            }
        },

        # ── 6. NUMÉRICOS ────────────────────────────────
        {
            "fix_numeric_columns": {
                "columns": [
                    "age",
                    "purchase_amount_(usd)",
                    "review_rating",
                    "previous_purchases"
                ],
                "strategy": "median"
            }
        },

        # ── 7. OUTLIERS ─────────────────────────────────
        {
            "cap_outliers": {
                "columns": [
                    "purchase_amount_(usd)",
                    "previous_purchases"
                ],
                "lower_percentile": 0.01,
                "upper_percentile": 0.99
            }
        },

        # ── 8. TIPOS FINALES ────────────────────────────
        {
            "fix_columns_types": {
                "numeric_columns": [
                    "age",
                    "purchase_amount_(usd)",
                    "review_rating",
                    "previous_purchases"
                ],
                "bool_columns": [
                    "discount_applied",
                    "promo_code_used",
                    "subscription_status"
                ],
                "date_columns": []
            }
        },
    ]
)
    result = pipeline.run(data)
    # ── MACHINE LEARNING preparation ─────────────────────────────
    # Only numeric/categorical columns that the model uses
    feature_columns = [
    "age",
    "review_rating",
    "previous_purchases",
    "gender_encoded",
    "category_encoded",
    "season_encoded",
]
    #NOTE:"""IMPORTANT: The standard scaler is not working as expected, it is not scaling the data properly"""
    previous_ml_pipeline = PipelineBuilder(result).build(
    configuration=[
        {"standard_scaler": {
            "columns": feature_columns
        }}
    ]
)
    result_scaled = previous_ml_pipeline.run(result)

    trend_analyzer = AnalyzerFactory.create("trend_patterns", result_scaled)
    result_trend = trend_analyzer.analyze(
        x="gender",
        y="purchase_amount_(usd)",
        type_of_analysis="gradient_descent",
        complexity="multiple",
        type_of_encoder="one_hot",
    )
    print(result_trend)

    seasonality_analyzer = AnalyzerFactory.create("seasonality", result)
    result_seasonality = seasonality_analyzer.analyze(
        target_column="purchase_amount_(usd)", period=12
    )
    print(result_seasonality)


if __name__ == "__main__":
    main()
