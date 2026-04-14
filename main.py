"""Main module for running the EDA project"""


from readers.reader_factory import ReaderFactory
from data_cleaning.data_cleaning_pipeline import build_pipeline_from_preset
from parsers.parser import parser

def main():
    """Main function for running the EDA project"""
    args = parser("data.csv")

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
