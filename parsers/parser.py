"""Module for parsing arguments"""
import argparse

def parser (file_path : str ,
 preset : str = "default"
 ) -> None:
    """Parse the arguments"""
    parser_object = argparse.ArgumentParser(description="Run data cleaning pipeline")
    #input
    parser_object.add_argument("-i",
    "--input",
    type=str,
    default=file_path,
    help="Input file path")
    #output
    parser_object.add_argument("-o",
    "--output", 
    type=str,
    help="Output file path (default: clean_<input>)")
    #preset
    parser_object.add_argument("-p",
    "--preset", 
    type=str,
    choices=["light", "default", "strict"],
    default=preset,
    help="Pipeline preset configuration")
    #report
    parser_object.add_argument("-r",
    "--report", 
    type=str,
    help="Path to save the JSON report (default: cleaning_report_<input>.json)")
    args = parser_object.parse_args()
    return args
