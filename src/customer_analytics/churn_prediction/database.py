import numpy as np
import pandas as pd
import re

from sqlalchemy import text
from sqlalchemy.engine.base import Engine


def get_schema_table_names(
    engine: Engine
) -> list[str]:

    with engine.connect() as conn:
        
        result = conn.execute(
            text("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'public';
            """)
        )

        table_names = result.fetchall()
        
    return [table_name[0] for table_name in table_names]


def create_summary_dataframes_from_tables(
    engine: Engine, 
    table_names: list[str]
) -> dict[str, pd.DataFrame] :
    """
    Creates a dictionary object with table names for key and corresponding summary dataframes as values.
    """
    summary_dfs = {}
    numeric_types = {
        "smallint",
        "integer",
        "bigint",
        "numeric",
        "decimal",
        "real",
        "double precision",
    }

    with engine.connect() as conn:

        for table_name in table_names:

            columns_df = pd.read_sql_query(
                text("""
                    SELECT
                        column_name,
                        data_type
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """),
                conn,
                params={"table_name": table_name},
            )

            summaries = []

            preparer = conn.dialect.identifier_preparer
            quoted_table = preparer.quote(table_name)

            for _, column in columns_df.iterrows():

                col = column["column_name"]
                dtype = column["data_type"]
                is_numeric = dtype in numeric_types

                quoted_col = preparer.quote(col)

                query = f"""
                    SELECT
                        '{col.replace("'", "''")}' AS column_name,
                        '{dtype.replace("'", "''")}' AS data_type,
                        COUNT({quoted_col}) AS count,
                        COUNT(*) - COUNT({quoted_col}) AS null_count,
                        COUNT(DISTINCT {quoted_col}) AS unique_count,
                        MIN({quoted_col}) AS min,
                        MAX({quoted_col}) AS max
                """

                if is_numeric:
                    query += f""",
                        AVG({quoted_col}) AS mean,
                        VAR_SAMP({quoted_col}) AS sample_variance,
                        STDDEV_SAMP({quoted_col}) AS sample_std
                    """
                else:
                    query += """,
                        NULL::double precision AS mean,
                        NULL::double precision AS sample_variance,
                        NULL::double precision AS sample_std
                    """

                query += f" FROM {quoted_table}"

                result = pd.read_sql_query(text(query), conn)
                summaries.append(result)

            if summaries:
                summary_dfs[table_name] = pd.concat(
                    summaries,
                    ignore_index=True,
                )
            else:
                summary_dfs[table_name] = pd.DataFrame()

    return summary_dfs


def create_dummy_columns(
    conn, 
    categorical_columns
):
    """
    Dynamically generate one dummy variable for every unique
    value in each categorical column, including NULL values.
    """

    dummy_expressions = []
    params = {}

    for column in categorical_columns:

        categories = conn.execute(
            text(f'''
                SELECT "{column}"
                FROM customers
                GROUP BY "{column}"
                ORDER BY "{column}"::text NULLS FIRST
            ''')
        ).scalars().all()

        used_names = set()

        for i, category in enumerate(categories):

            if category is None:
                category_name = "missing"
            else:
                category_name = re.sub(
                    r"[^a-zA-Z0-9]+",
                    "_",
                    str(category).lower()
                ).strip("_") or "empty"

            dummy_name = f"{column.lower()}__{category_name}"

            if dummy_name in used_names:
                dummy_name = f"{dummy_name}_{i}"

            used_names.add(dummy_name)

            param_name = f"dummy_{column}_{i}"

            dummy_expressions.append(
                f'''
                CASE
                    WHEN c."{column}" IS NOT DISTINCT FROM :{param_name}
                    THEN 1
                    ELSE 0
                END AS "{dummy_name}"
                '''
            )

            params[param_name] = category

    return ",\n".join(dummy_expressions), params


class ClassifierDataIterator:

    def __init__(
        self,
        engine,
        split,
        batch_size=10_000,
        start_date=None,
        end_date=None
    ):

        valid_splits = {"train", "validation", "test"}

        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.engine = engine
        self.split = split
        self.batch_size = batch_size
        self.start_date = start_date
        self.end_date = end_date

        self.conn = None
        self.generator = None

    def _close(self):

        if self.generator is not None:
            self.generator.close()
            self.generator = None

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def reset(self):

        self._close()

        self.conn = self.engine.connect().execution_options(
            stream_results=True,
            yield_per=self.batch_size
        )

        query = text("""
            SELECT *
            FROM churn_model_dataset
            WHERE dataset_split = :dataset_split
                AND age IS NOT NULL
                AND (
                    CAST(:start_date AS DATE) IS NULL
                    OR cutoff_date >= CAST(:start_date AS DATE)
                )
                AND (
                    CAST(:end_date AS DATE) IS NULL
                    OR cutoff_date < CAST(:end_date AS DATE)
                )
            ORDER BY cutoff_date
        """)

        try:
            chunks = pd.read_sql_query(
                query,
                self.conn,
                params={
                    "dataset_split": self.split,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                chunksize=self.batch_size
            )

            self.generator = self._process_batches(chunks)

        except Exception:
            self._close()
            raise

    def _process_batches(self, chunks):

        columns_to_drop = [
            "club_member_status__missing",
            "customer_id",
            "cutoff_date",
            "dataset_split",
            "fashion_news_frequency__none"
        ]

        for df in chunks:

            y = df["churned"].to_numpy(dtype=np.float32)

            X = (
                df.drop(
                    columns=columns_to_drop + ["churned"]
                )
                .to_numpy(dtype=np.float32)
            )

            yield X, y


    def get_batches(self):

        self.reset()

        try:
            yield from self.generator

        finally:
            self._close()