"""hnsw index on chunks.embedding

Revision ID: c8dcd3c6bb84
Revises: 4fc4cb4f9bdb
Create Date: 2026-08-20 04:02:26.837863

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8dcd3c6bb84"
down_revision: str | Sequence[str] | None = "4fc4cb4f9bdb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks "
        "USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX chunks_embedding_idx;")
