from clients import get_db


def test_db_connection():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT 1 AS result")
        row = cur.fetchone()
    print("DB connection OK:", row)
    assert row["result"] == 1


if __name__ == "__main__":
    test_db_connection()
    print("All DB tests passed.")