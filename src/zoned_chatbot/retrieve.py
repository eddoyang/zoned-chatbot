import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import DATABASE_URL, PER_DOC_CAP, POOL_SIZE, TOP_K
from .embed import embed_query

SQL = """
WITH pool AS (
    SELECT c.id, c.doc_id, c.content, c.page,
            c.embedding <=> %(qvec)s AS distance
    FROM chunks c
    WHERE %(doc_ids)s::bigint[] IS NULL OR c.doc_id = ANY(%(doc_ids)s)
    ORDER BY c.embedding <=> %(qvec)s
    LIMIT %(pool)s
),
ranked AS (
    SELECT p.*,
            ROW_NUMBER() OVER (PARTITION BY p.doc_id ORDER BY p.distance) AS per_doc
    FROM pool p
)
SELECT r.id, r.content, r.page, r.distance, d.title, d.filename
FROM ranked r
JOIN documents d ON d.id = r.doc_id
WHERE r.per_doc <= %(cap)s
ORDER BY r.distance
LIMIT %(k)s;
"""


def retrieve(
    question: str, k: int = TOP_K, doc_ids: list[int] | None = None
) -> list[dict]:
    cap = k if doc_ids and len(doc_ids) == 1 else PER_DOC_CAP
    qvec = Vector(embed_query(question))
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                SQL,
                {
                    "qvec": qvec,
                    "pool": POOL_SIZE,
                    "cap": cap,
                    "k": k,
                    "doc_ids": doc_ids,
                },
            )
            return cur.fetchall()
