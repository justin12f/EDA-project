"""Main module for running the EDA project"""

import argparse
from readers.reader_factory import ReaderFactory
from data_cleaning.data_cleaning_pipeline import build_pipeline_from_preset


def main():
    """Main function for running the EDA project"""
    parser = argparse.ArgumentParser(description="Run data cleaning pipeline")
    parser.add_argument("-i", "--input", type=str, default="dirty_data.csv", help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path (default: clean_<input>)")
    parser.add_argument("-p", "--preset", type=str, choices=["light", "default", "strict"], default="default", help="Pipeline preset configuration")
    parser.add_argument("-r", "--report", type=str, help="Path to save the JSON report (default: cleaning_report_<input>.json)")
    args = parser.parse_args()

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

    print("\n" + "="*100)
    print(" CLEANED DATA PREVIEW ")
    print("="*100)
    print(result.head().to_string())
    print("\n")
    pipeline.report.print_summary()
    print(result)


if __name__ == "__main__":
    main()
