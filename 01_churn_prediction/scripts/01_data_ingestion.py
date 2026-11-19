
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv()

engine = create_engine(
    os.environ["DATABASE_URL"],
    pool_pre_ping=True,
)

for csv_path in sorted(DATA_DIR.glob("*.csv")):
    table_name = csv_path.stem.lower()
    first_chunk = True

    print(f"Loading {csv_path.name} into {table_name}...")

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        chunk.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=1_000,
        )
        first_chunk = False

    print(f"Finished loading {table_name}")
    
with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE transactions
        ALTER COLUMN t_dat TYPE TIMESTAMP
        USING t_dat::timestamp;
    """))

engine.dispose()