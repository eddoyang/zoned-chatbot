"""initial schema

Revision ID: 4fc4cb4f9bdb
Revises: 
Create Date: 2026-08-20 00:34:23.011751

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4fc4cb4f9bdb'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.execute("""
        CREATE TABLE documents (
            id            BIGSERIAL PRIMARY KEY,
            title         TEXT NOT NULL,
            filename      TEXT NOT NULL,
            content_hash  TEXT NOT NULL UNIQUE,
            page_count    INT,
            ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)


    op.execute("""
        CREATE TABLE chunks (
            id            BIGSERIAL PRIMARY KEY,
            doc_id        BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index   INT NOT NULL,
            content       TEXT NOT NULL,
            heading       TEXT,
            page          INT,
            char_start    INT,
            char_end      INT,
            chunk_type    TEXT NOT NULL DEFAULT 'prose',
            embedding     VECTOR(1536),
            tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            UNIQUE (doc_id, chunk_index)
        );
    """)

    op.execute("CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);")
    op.execute("CREATE INDEX chunks_doc_idx ON chunks (doc_id);")


def downgrade() -> None:
    op.execute("DROP TABLE chunks;")
    op.execute("DROP TABLE documents;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
