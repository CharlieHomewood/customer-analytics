import os

from customer_analytics.database import create_dummy_columns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

categorical_columns = [
    "club_member_status",
    "fashion_news_frequency",
]

with engine.begin() as conn:

    dummy_sql, dummy_params = create_dummy_columns(
        conn,
        categorical_columns
    )

    conn.execute(
        text("DROP TABLE IF EXISTS churn_model_dataset")
    )

    conn.execute(
        text(f"""
            CREATE TABLE churn_model_dataset AS

            WITH customer_buckets AS (
                SELECT
                    customer_id,
                    MOD(
                        ABS(hashtextextended(customer_id, 42)),
                        10
                    ) AS customer_bucket
                FROM customers
            ),

            snapshots AS (
                SELECT
                    (d - INTERVAL '1 day')::date AS cutoff_date,

                    CASE
                        WHEN d < DATE '2020-05-01'
                            THEN 'train'

                        WHEN d < DATE '2020-06-01'
                            THEN 'validation'

                        ELSE 'test'
                    END AS dataset_split,

                    c.customer_id

                FROM generate_series(
                    DATE '2019-10-01',
                    DATE '2020-07-01',
                    INTERVAL '1 month'
                ) AS g(d)

                CROSS JOIN customers c
            ),

            features AS (
                SELECT
                    s.customer_id,
                    s.cutoff_date,
                    s.dataset_split,

                    SUM(a.items_purchased)
                        AS items_last_12_months,

                    COALESCE(
                        SUM(a.items_purchased) FILTER (
                            WHERE a.transaction_date >
                                s.cutoff_date - INTERVAL '30 days'
                        ),
                        0
                    ) AS items_last_30_days,

                    COUNT(*) FILTER (
                        WHERE a.transaction_date >
                            s.cutoff_date - INTERVAL '90 days'
                    ) AS purchase_days_last_90_days,

                    COUNT(*)
                        AS purchase_days_last_12_months,

                    SUM(a.daily_spend)
                        AS spend_last_12_months,

                    MAX(a.transaction_date)
                        AS last_purchase_date,

                    s.cutoff_date - MAX(a.transaction_date)
                        AS days_since_last_purchase,

                    SUM(a.distinct_articles)
                        AS article_day_count

                FROM snapshots s

                JOIN customer_daily_activity a
                    ON a.customer_id = s.customer_id
                    AND a.transaction_date >
                        s.cutoff_date - INTERVAL '12 months'
                    AND a.transaction_date <= s.cutoff_date

                GROUP BY
                    s.customer_id,
                    s.cutoff_date,
                    s.dataset_split
            ),

            future_activity AS (
                SELECT
                    s.customer_id,
                    s.cutoff_date,

                    COUNT(a.customer_id)
                        AS future_purchase_days

                FROM snapshots s

                LEFT JOIN customer_daily_activity a
                    ON a.customer_id = s.customer_id
                    AND a.transaction_date > s.cutoff_date
                    AND a.transaction_date < (
                        DATE_TRUNC('month', s.cutoff_date)
                        + INTERVAL '2 months'
                    )::date

                GROUP BY
                    s.customer_id,
                    s.cutoff_date
            )

            SELECT
                f.customer_id,
                f.cutoff_date,
                f.dataset_split,

                c.age,

                {dummy_sql},

                f.items_last_12_months,
                f.items_last_30_days,
                f.purchase_days_last_90_days,
                f.purchase_days_last_12_months,
                f.spend_last_12_months,
                f.days_since_last_purchase,
                f.article_day_count,

                CASE
                    WHEN fa.future_purchase_days = 0
                    THEN 1
                    ELSE 0
                END AS churned

            FROM features f

            JOIN customers c
                ON c.customer_id = f.customer_id

            JOIN future_activity fa
                ON fa.customer_id = f.customer_id
                AND fa.cutoff_date = f.cutoff_date;
        """),
        dummy_params
    )

    conn.execute(
        text("""
            CREATE INDEX idx_churn_dataset_split
            ON churn_model_dataset (dataset_split)
        """)
    )

    conn.execute(
        text("ANALYZE churn_model_dataset")
    )