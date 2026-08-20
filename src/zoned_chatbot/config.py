import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CORPUS_ROOT = ROOT / "corpus"
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100