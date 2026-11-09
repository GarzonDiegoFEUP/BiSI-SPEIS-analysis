#!/usr/bin/env python3
"""Convert a tab-delimited BioLogic EC-Lab .mpt file to CSV, row by row."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from itertools import islice
from typing import TextIO

EIS_COLUMNS = (
    "freq/Hz",
    "Re(Z)/Ohm",
    "-Im(Z)/Ohm",
    "|Z|/Ohm",
    "Phase(Z)/deg",
    "time/s",
    "<Ewe>/V",
    "<I>/mA",
    "cycle number",
)


def find_table_header(source: TextIO) -> tuple[list[str], int]:
    """Return the table column names and the 1-based header line number."""
    for line_number, line in enumerate(source, start=1):
        if line.lstrip("\ufeff").startswith("freq/Hz"):
            columns = [column.strip() for column in line.rstrip("\r\n").split("\t")]
            # Some EC-Lab exports end the header with a tab, creating an empty
            # final field that has no corresponding value in the data rows.
            while columns and not columns[-1]:
                columns.pop()
            if not columns:
                raise ValueError(f"No columns found on header line {line_number}.")
            return columns, line_number
    raise ValueError("Could not find the tab-separated 'freq/Hz' table header.")


def convert(
    input_path: Path,
    output_path: Path,
    *,
    all_columns: bool = False,
    preview_rows: int | None = None,
) -> int:
    """Write the measurement table to CSV and return the number of data rows."""
    with input_path.open("r", encoding="cp1252", newline="") as source:
        columns, header_line = find_table_header(source)
        if all_columns:
            selected_indices = list(range(len(columns)))
            output_columns = columns
        else:
            missing = [name for name in EIS_COLUMNS if name not in columns]
            if missing:
                raise ValueError(
                    "Required EIS columns are missing: " + ", ".join(missing)
                )
            selected_indices = [columns.index(name) for name in EIS_COLUMNS]
            output_columns = list(EIS_COLUMNS)

        with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(output_columns)
            row_count = 0

            data_lines = enumerate(source, start=header_line + 1)
            if preview_rows is not None:
                data_lines = islice(data_lines, preview_rows)

            for line_number, line in data_lines:
                if not line.strip():
                    continue

                values = [value.strip() for value in line.rstrip("\r\n").split("\t")]
                while values and not values[-1]:
                    values.pop()
                if len(values) != len(columns):
                    raise ValueError(
                        f"Line {line_number} has {len(values)} fields; "
                        f"expected {len(columns)} based on the table header."
                    )
                writer.writerow([values[index] for index in selected_indices])
                row_count += 1

    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a BioLogic EC-Lab .mpt measurement table to CSV."
    )
    parser.add_argument("input", type=Path, help="Path to the source .mpt file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="CSV output path (default: same path as input with a .csv extension)",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Export every measurement column instead of the compact EIS subset",
    )
    parser.add_argument(
        "--preview",
        type=int,
        metavar="ROWS",
        help="Export only the first ROWS measurement rows for a small preview",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".csv")
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        parser.error("Input and output paths must be different.")
    if args.preview is not None and args.preview < 1:
        parser.error("--preview must be a positive number of rows.")

    try:
        rows = convert(
            input_path,
            output_path,
            all_columns=args.all_columns,
            preview_rows=args.preview,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {rows:,} data rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
