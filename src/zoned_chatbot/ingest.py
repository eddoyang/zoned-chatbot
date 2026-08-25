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


def ingest(path: Path) -> None:
    rel = str(path.relative_to(CORPUS_ROOT))
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    parsed = parse(path)
    chunks = chunk(parsed)
    vectors = embed_texts([c.content for c in chunks])

    assert len(chunks) == len(vectors), (
        f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
    )

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (title, filename, content_hash, page_count)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (path.stem, rel, content_hash, parsed.page_count),
            )

            doc_id = cur.fetchone()[0]

            cur.executemany(
                """
                INSERT INTO chunks
                (doc_id, chunk_index, content, page, char_start, char_end, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        doc_id,
                        c.index,
                        c.content,
                        c.page,
                        c.char_start,
                        c.char_end,
                        Vector(v),
                    )
                    for c, v in zip(chunks, vectors)
                ],
            )

    print(f"ingested {rel}: {len(chunks)} chunks, {parsed.page_count} pages")


if __name__ == "__main__":
    ingest(Path(sys.argv[1]).resolve())
