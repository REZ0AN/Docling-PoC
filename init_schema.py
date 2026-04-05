import os
import logging
import psycopg2
from clients import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")


def run_schema(schema_file: str = SCHEMA_FILE) -> None:
    """
    Apply schema.sql to the NeonDB database.
    Uses autocommit=True so CREATE EXTENSION runs outside a transaction block.
    Safe to re-run if schema.sql uses IF NOT EXISTS guards.

    Note: bypasses DBClient.cursor() here because CREATE EXTENSION requires
    autocommit=True, which must be set directly on the underlying connection
    before any statement is executed.
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, "r") as f:
        sql = f.read()

    if not sql.strip():
        raise ValueError(f"Schema file is empty: {schema_file}")

    try:
        db = get_db()
        conn = db.conn
        conn.rollback()        # clear any open transaction before changing autocommit
        conn.autocommit = True  # required for CREATE EXTENSION

        logger.info("Applying schema from: %s", schema_file)
        with conn.cursor() as cur:
            cur.execute(sql)

        logger.info("Schema applied successfully.")

    except psycopg2.Error as e:
        logger.error("Database error while applying schema: %s", e)
        raise
    finally:
        # Reset autocommit so the shared singleton connection
        # behaves normally for all subsequent operations.
        conn.autocommit = False
        logger.info("autocommit reset to False.")


if __name__ == "__main__":
    run_schema()