import hashlib
import sys
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from .chunk import chunk
from .config import CORPUS_ROOT, DATABASE_URL
from .embed import embed_texts
from .parse import parse


def ingest(path: Path) -> None:
    rel = str(path.relative_to(CORPUS_ROOT))
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    parsed = parse(path)
    chunks = chunk(parsed)
    vectors = embed_texts([c.content for c in chunks])

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
