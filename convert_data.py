#!/usr/bin/env python3
"""
Data format converter utility for ILA Machine Learning project.
Converts between CSV and Excel (XLSX) formats.
"""

import pandas as pd
import sys
import os
from pathlib import Path


def csv_to_excel(csv_file: str, excel_file: str = None):
    """Convert CSV file to Excel XLSX format."""
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return False

    if excel_file is None:
        # Generate Excel filename from CSV filename
        excel_file = Path(csv_file).with_suffix('.xlsx')

    try:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Write to Excel file
        df.to_excel(excel_file, index=False)

        print(f"✓ Converted: {csv_file} → {excel_file}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        return True

    except Exception as e:
        print(f"Error converting {csv_file}: {e}")
        return False


def excel_to_csv(excel_file: str, csv_file: str = None):
    """Convert Excel XLSX file to CSV format."""
    if not os.path.exists(excel_file):
        print(f"Error: Excel file '{excel_file}' not found.")
        return False

    if csv_file is None:
        # Generate CSV filename from Excel filename
        csv_file = Path(excel_file).with_suffix('.csv')

    try:
        # Read Excel file
        df = pd.read_excel(excel_file)

        # Write to CSV file
        df.to_csv(csv_file, index=False)

        print(f"✓ Converted: {excel_file} → {csv_file}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        return True

    except Exception as e:
        print(f"Error converting {excel_file}: {e}")
        return False


def batch_convert_csv_to_excel(directory: str = "."):
    """Convert all CSV files in directory to Excel format."""
    csv_files = list(Path(directory).glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Found {len(csv_files)} CSV files to convert:")
    for csv_file in csv_files:
        excel_file = csv_file.with_suffix('.xlsx')
        csv_to_excel(str(csv_file), str(excel_file))


def batch_convert_excel_to_csv(directory: str = "."):
    """Convert all Excel files in directory to CSV format."""
    excel_files = list(Path(directory).glob("*.xlsx")) + \
        list(Path(directory).glob("*.xls"))

    if not excel_files:
        print(f"No Excel files found in {directory}")
        return

    print(f"Found {len(excel_files)} Excel files to convert:")
    for excel_file in excel_files:
        csv_file = excel_file.with_suffix('.csv')
        excel_to_csv(str(excel_file), str(csv_file))


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert single file:")
        print(
            "    python convert_data.py csv-to-excel <input.csv> [output.xlsx]")
        print(
            "    python convert_data.py excel-to-csv <input.xlsx> [output.csv]")
        print("  Batch convert:")
        print("    python convert_data.py batch-csv-to-excel [directory]")
        print("    python convert_data.py batch-excel-to-csv [directory]")
        print("  Examples:")
        print("    python convert_data.py csv-to-excel training_data.csv")
        print("    python convert_data.py excel-to-csv training_data.xlsx")
        print("    python convert_data.py batch-csv-to-excel")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "csv-to-excel":
        if len(sys.argv) < 3:
            print("Error: Please provide CSV file path")
            sys.exit(1)

        csv_file = sys.argv[2]
        excel_file = sys.argv[3] if len(sys.argv) > 3 else None
        csv_to_excel(csv_file, excel_file)

    elif command == "excel-to-csv":
        if len(sys.argv) < 3:
            print("Error: Please provide Excel file path")
            sys.exit(1)

        excel_file = sys.argv[2]
        csv_file = sys.argv[3] if len(sys.argv) > 3 else None
        excel_to_csv(excel_file, csv_file)

    elif command == "batch-csv-to-excel":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        batch_convert_csv_to_excel(directory)

    elif command == "batch-excel-to-csv":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        batch_convert_excel_to_csv(directory)

    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
