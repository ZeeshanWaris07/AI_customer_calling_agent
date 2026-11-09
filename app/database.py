
import json
import os

from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def load_json(filename: str):
    file_path = DATA_DIR / filename

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_faqs():
    return load_json("faqs.json")


def load_orders():
    return load_json("orders.json")


def get_db_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing from the .env file."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=10,
    )


def initialize_database() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                estimated_delivery DATE
            )
            """
        )


def seed_orders() -> int:
    orders = load_orders()

    if not orders:
        return 0

    initialize_database()

    with get_db_connection() as connection:
        for order in orders:

            estimated_delivery = order.get(
                "estimated_delivery"
            )

            if estimated_delivery:
                estimated_delivery = date.fromisoformat(
                    estimated_delivery
                )

            connection.execute(
                """
                INSERT INTO orders (
                    order_id,
                    customer_id,
                    status,
                    estimated_delivery
                )
                VALUES (%s, %s, %s, %s)

                ON CONFLICT (order_id)
                DO UPDATE SET
                    customer_id = EXCLUDED.customer_id,
                    status = EXCLUDED.status,
                    estimated_delivery = EXCLUDED.estimated_delivery
                """,
                (
                    order["order_id"],
                    order["customer_id"],
                    order["status"],
                    estimated_delivery,
                ),
            )

    return len(orders)


def get_order_by_id(order_id: str) -> dict | None:
    with get_db_connection() as connection:

        order = connection.execute(
            """
            SELECT
                order_id,
                customer_id,
                status,
                estimated_delivery
            FROM orders
            WHERE order_id = %s
            """,
            (order_id,),
        ).fetchone()

    return order
