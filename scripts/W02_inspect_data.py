
from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

tables = [
    "sites",
    "robots",
    "telemetry",
    "missions",
    "faults",
    "maintenance",
]

for table in tables:
    file_path = RAW_DIR / f"{table}.csv"

    print(f"\n{'=' * 50}")
    print(f"TABLE: {table.upper()}")
    print(f"{'=' * 50}")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns and data types:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nExact duplicate rows:")
    print(df.duplicated().sum())
