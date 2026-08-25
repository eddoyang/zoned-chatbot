import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunk import chunk
from .config import CORPUS_ROOT, DATABASE_URL
from .embed import embed_texts
from .parse import parse

SQL_FIND_BY_HASH = """
    SELECT id FROM documents WHERE content_hash = %(content_hash)s
"""

SQL_FIND_BY_FILENAME = """
    SELECT id, content_hash FROM documents WHERE filename = %(filename)s
"""

SQL_INSERT_DOC = """
    INSERT INTO documents (title, filename, content_hash, page_count)
    VALUES (%(title)s, %(filename)s, %(content_hash)s, %(page_count)s)
    ON CONFLICT (content_hash) DO NOTHING
    RETURNING id
"""

SQL_INSERT_CHUNKS = """
    INSERT INTO chunks 
    (doc_id, chunk_index, content, page, char_start, char_end, embedding)
    VALUES (%(doc_id)s, %(chunk_index)s, %(content)s, %(page)s, %(char_start)s, %(char_end)s, %(embedding)s)
"""

SQL_DELETE_DOC = """
    DELETE FROM documents WHERE id = %(id)s
"""

SQL_LIST = """
    SELECT d.filename, d.page_count, count(c.id) AS chunks,
            min(c.page) AS lo, max(c.page) AS hi, d.ingested_at
    FROM documents d
    LEFT JOIN chunks c ON c.doc_id = d.id
    GROUP BY d.id, d.filename, d.page_count, d.ingested_at
    ORDER BY d.filename
"""


@dataclass
class Result:
    rel: str
    status: str
    detail: str = ""


def ingest_one(conn: psycopg.Connection, path: Path, replace: bool = False) -> Result:
    rel = str(path.resolve().relative_to(CORPUS_ROOT.resolve()))
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    stale_id: int | None = None


    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SQL_FIND_BY_HASH, {"content_hash": content_hash})
        row = cur.fetchone()
        if row is not None:
            return Result(rel, "skipped", "already ingested")

        cur.execute(SQL_FIND_BY_FILENAME, {"filename": rel})
        rows = cur.fetchall()
        if len(rows) > 1:
            return Result(rel, "failed", f"{len(rows)} rows share this filename")
        
        if rows:
            if not replace:
                return Result(rel, "changed", "content changed on disk; re-run with --replace")

            stale_id = rows[0]["id"]


    parsed = parse(path)
    chunks = chunk(parsed)
    vectors = embed_texts([c.content for c in chunks])

    with conn.transaction(), conn.cursor() as cur:

        if stale_id is not None:
            cur.execute(SQL_DELETE_DOC, {"id": stale_id})

        
        cur.execute(
            SQL_INSERT_DOC,
            {
                "title": path.stem,
                "filename": rel,
                "content_hash": content_hash,
                "page_count": parsed.page_count,
            },
        )
        

        row = cur.fetchone(row_factory=dict_row)
        if row is None:
            raise RuntimeError(f"{rel}: insert conflicted after hash check")
        
        doc_id = row["id"]

        cur.executemany(
            SQL_INSERT_CHUNKS,
            [       
                {
                    "doc_id": doc_id,
                    "chunk_index": c.index,
                    "content": c.content,
                    "page": c.page,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "embedding": Vector(v),
                }
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
        
    status = "replaced" if stale_id is not None else "ingested"
    return Result(rel, status, f"{len(chunks)} chunks, {parsed.page_count} pages")


if __name__ == "__main__":
    pass