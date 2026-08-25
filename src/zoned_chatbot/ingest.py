import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunk import chunk
from .config import DATABASE_URL
from .embed import embed_texts
from .parse import parse

SQL_FIND_BY_HASH = """
    SELECT id, filename FROM documents WHERE content_hash = %(content_hash)s
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
    SELECT d.id, d.filename, d.page_count, count(c.id) AS chunks,
            min(c.page) AS lo, max(c.page) AS hi, d.ingested_at
    FROM documents d
    LEFT JOIN chunks c ON c.doc_id = d.id
    GROUP BY d.id
    ORDER BY d.filename
"""

SQL_UPDATE_FILENAME = """
    UPDATE documents SET filename = %(filename)s WHERE id = %(id)s
"""


@dataclass
class Result:
    rel: str
    status: str
    detail: str = ""


def ingest_one(
    conn: psycopg.Connection, path: Path, root: Path, replace: bool = False
) -> Result:
    rel = str(path.resolve().relative_to(root.resolve()))
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    stale_id: int | None = None

    with conn.cursor() as cur:
        cur.execute(SQL_FIND_BY_HASH, {"content_hash": content_hash})
        row = cur.fetchone()

        if row is not None:
            # Same content, same path; already exists
            if row["filename"] == rel:
                return Result(rel, "skipped", "already ingested")

            # Same content, different path, old file still on disk; duplicate
            old = root / row["filename"]
            if old.exists():
                return Result(rel, "failed", f"identical content to {row['filename']}")

            # Same content, different path, old file gone; rename
            cur.execute(SQL_UPDATE_FILENAME, {"id": row["id"], "filename": rel})
            return Result(rel, "renamed", f"was {row['filename']}")

        cur.execute(SQL_FIND_BY_FILENAME, {"filename": rel})
        rows = cur.fetchall()

        if len(rows) > 1:
            return Result(rel, "failed", f"{len(rows)} rows share this filename")

        if rows:
            if not replace:
                return Result(
                    rel,
                    "changed",
                    "file differs from ingested version; re-run with --replace",
                )

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

        row = cur.fetchone()
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


def ingest_dir(root: Path, replace: bool = False, force: bool = False) -> list[Result]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    results: list[Result] = []

    paths = sorted(root.rglob("*.pdf"))

    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)

        if force:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE documents CASCADE")

        for path in paths:
            rel = path.resolve().relative_to(root)

            try:
                # Skip any deferred documents for now.
                if "deferred" in rel.parts:
                    results.append(Result(str(rel), "deferred"))
                else:
                    results.append(ingest_one(conn, path, root, replace=replace))

            except Exception as exc:  # noqa: BLE001
                results.append(
                    Result(str(rel), "failed", f"{type(exc).__name__}: {exc}")
                )

            print(f"{results[-1].status:9} {results[-1].rel} {results[-1].detail}")

    return results


def summarize(results: list[Result]) -> None:
    counts = Counter(r.status for r in results)
    print("  ·  ".join(f"{status} {n}" for status, n in sorted(counts.items())))

    if counts["failed"]:
        sys.exit(1)


def list_docs() -> None:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(SQL_LIST)
            rows = cur.fetchall()

    if not rows:
        print("No documents ingested")
        return

    print(f"{'id':>4}  {'filename':<32} {'pages':>6} {'chunks':>7} {'lo':>4} {'hi':>5}")
    for r in rows:
        print(
            f"{r['id']:>4}  {r['filename']:<32} {r['page_count'] or 0:>6} "
            f"{r['chunks']:>7} {r['lo'] or 0:>4} {r['hi'] or 0:>5}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(prog="zoned-ingest")
    ap.add_argument("path", nargs="?", type=Path, help="PDF or directory")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="re-ingest documents whose content has changed",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="drop everything and re-ingest from scratch",
    )
    ap.add_argument("--list", action="store_true", help="show what's ingested")
    args = ap.parse_args()

    if args.list:
        list_docs()
        return

    if args.path is None:
        ap.error("path is required unless --list")

    if args.force and not args.path.is_dir():
        ap.error("--force only applies to a directory")

    if args.path.is_dir():
        summarize(ingest_dir(args.path, replace=args.replace, force=args.force))
    else:
        root = args.path.resolve().parent
        with psycopg.connect(
            DATABASE_URL, autocommit=True, row_factory=dict_row
        ) as conn:
            register_vector(conn)
            result = ingest_one(conn, args.path, root, replace=args.replace)

        print(f"{result.status:9} {result.rel} {result.detail}")


if __name__ == "__main__":
    main()
