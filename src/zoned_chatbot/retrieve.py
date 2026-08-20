import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import DATABASE_URL, TOP_K
from .embed import embed_query

SQL = """
SELECT c.id, c.content, c.page, d.title, d.filename,
    c.embedding <=> %(qvec)s AS distance
FROM chunks c
JOIN documents d ON d.id = c.doc_id
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s;
"""


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    qvec = Vector(embed_query(question))
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(SQL, {"qvec": qvec, "k": k})
            return cur.fetchall()