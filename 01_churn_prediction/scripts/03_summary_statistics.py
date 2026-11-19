import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from customer_analytics.database import (
    create_summary_dataframes_from_tables, 
    get_schema_table_names
)


def main():


    DATA_SUMMARIES_DIR = "data/summaries"
    os.makedirs(DATA_SUMMARIES_DIR, exist_ok=True)


    load_dotenv()
    engine = create_engine(os.getenv("DATABASE_URL"))


    table_names = get_schema_table_names(engine=engine)
    
    summary_dfs = create_summary_dataframes_from_tables(
        engine=engine, 
        table_names=table_names
    )


    for table_name, df in summary_dfs.items():

        if df.empty:
            continue

        for column in ["column_name", "data_type"]:
            df[column] = df[column].astype("string")

        for column in ["count", "null_count", "unique_count"]:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).astype("Int64")

        for column in ["mean", "sample_variance", "sample_std"]:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).astype("Float64")

        for column in ["min", "max"]:
            df[column] = df[column].map(
                lambda x: str(x) if pd.notna(x) else None
            ).astype("object")

        df.to_parquet(
            f"{DATA_SUMMARIES_DIR}/{table_name}_summary.parquet",
            index=False,
            compression="zstd",
        )


if __name__ == "__main__":
    main()