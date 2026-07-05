from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras # RealDictCursor, execute_batch
from pgvector.psycopg2 import register_vector

from configs.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

def get_connection():
    kwargs_dict = {
        "host"      : DB_HOST,
        "port"      : DB_PORT, # 5432
        "dbname"    : DB_NAME,
        "user"      : DB_USER,
        "password"  : DB_PASSWORD,
    }

    return psycopg2.connect(**kwargs_dict)

@contextmanager
def db_connection(pool=None):
    conn = pool.getconn() if pool else get_connection()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn() if pool else conn.close()

def execute_query(sql: str, params: tuple | None=None, pool=None) -> list[str, Any]:
    "execute a SELECT queriy and return all rows as list[field, value]."
    with db_connection(pool=pool) as conn:
        with conn.cursor(cursor_factor = psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [row for row in cur.fetchall()]

def execute_write(sql: str, params: tuple | None=None, pool=None):
    "execute UPDATE/INSERT/DELETE queries."
    with db_connection(pool=pool) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)

def execute_batch(sql: str, params_list: list[tuple] | None=None, pool=None):
    "execute a write query for many rows."
    with db_connection(pool=pool) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list)

def init_db(schema_path: str = "./db/schema.sql"):
    with open(schema_path, "r") as f:
        sql = f.read()
    conn = get_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()

if __name__=="__main__":
    init_db()
