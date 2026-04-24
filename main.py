"""Main module for running the EDA project"""

from analyze_data.data_analyzer_factory import AnalyzerFactory
from data_cleaning.data_cleaning_pipeline import build_pipeline_from_preset
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
    pipeline = build_pipeline_from_preset(data, preset=args.preset)
    print("Running pipeline...")
    result = pipeline.run(data)

    print(f"Saving cleaned data to {args.output}...")
    result.to_csv(args.output, index=False)
    print(f"Saving report to {args.report}...")
    pipeline.report.to_json(args.report)
    print(result)

    print("\n" + "=" * 100)
    print(" CLEANED DATA PREVIEW ")
    print("=" * 100)
    print(result.head().to_string())
    print("\n")
    pipeline.report.print_summary()
    print(result)

    trend_analyzer = AnalyzerFactory.create("trend_patterns", result)
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
