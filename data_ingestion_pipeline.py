import json
import logging
import time
from pathlib import Path
from clients import get_gemini, get_db
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


ENRICH_PROMPT = """You are a document intelligence assistant.
Given the following text chunk, return a JSON object with:
- "summary": one/two sentence summary (max 50 words)
- "keywords": list of 3-6 key terms
- "hypo_qa": list of 3-5 hypothetical questions a user might ask
  that this chunk answers, each as {{"q": "...", "a": "..."}}

Respond with ONLY valid JSON, no markdown.

CHUNK:
{chunk_text}
"""

# ── Rate limit config ─────────────────────────────────────
ENRICH_BATCH_SIZE  = 13    # requests per batch  (limit: 15/min)
ENRICH_BATCH_WAIT  = 30    # seconds to pause after each batch

EMBED_BATCH_SIZE   = 98    # requests per batch  (limit: 100/min)
EMBED_BATCH_WAIT   = 30    # seconds to pause after each batch


# ── Stage 1: Restructure ──────────────────────────────────
def restructure(file_path: str):
    """Parse PDF/DOCX/HTML into a structured DoclingDocument."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document


# ── Stage 2: Chunk ────────────────────────────────────────
def chunk(doc) -> list:
    """
    Boundary-aware chunking — never splits mid-table or mid-code-block.
    Merges tiny consecutive chunks to reduce noise.
    """
    chunker = HybridChunker(max_tokens=512, merge_peers=True)
    return list(chunker.chunk(doc))


# ── Stage 3: Enrich ───────────────────────────────────────
def enrich(chunk_text: str) -> tuple[dict, dict]:
    """Generate summary, keywords, and hypothetical Q&A for a chunk.
    Returns (metadata, usage).
    """
    gemini = get_gemini()
    response, usage = gemini.generate(
        ENRICH_PROMPT.format(chunk_text=chunk_text[:3000]),
        json_mode=True
    )
    cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned), usage


# ── Stage 4: Embed + Store ────────────────────────────────
def store(doc_id: str, chunk_index: int, chunk_text: str,
          metadata: dict, enrich_usage: dict,
          dept: str = None, doc_type: str = None):
    """Embed a chunk, insert it into chunks, and log token usage."""
    gemini = get_gemini()
    db = get_db()

    vector, embed_usage = gemini.embed(chunk_text)

    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO chunks
                (doc_id, chunk_index, content, summary, keywords,
                 hypo_qa, dept, doc_type, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            doc_id, chunk_index, chunk_text,
            metadata.get("summary"),
            metadata.get("keywords", []),
            json.dumps(metadata.get("hypo_qa", [])),
            dept, doc_type, vector,
        ))
        chunk_id = cur.fetchone()["id"]

        cur.execute("""
            INSERT INTO token_usage
                (chunk_id, doc_id, stage, model, input_tokens, output_tokens)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (chunk_id, doc_id, "enrich",
              enrich_usage["model"],
              enrich_usage["input_tokens"],
              enrich_usage["output_tokens"]))

        cur.execute("""
            INSERT INTO token_usage
                (chunk_id, doc_id, stage, model, input_tokens, output_tokens)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (chunk_id, doc_id, "embed",
              embed_usage["model"],
              embed_usage["input_tokens"],
              embed_usage["output_tokens"]))

    db.commit()


# ── Rate-limit helpers ────────────────────────────────────
def _wait(seconds: int, reason: str):
    logger.info("Rate-limit pause (%s): waiting %ds ...", reason, seconds)
    time.sleep(seconds)


def process_chunks(
    doc_id: str,
    indexed_texts: list[tuple[int, str]],
    dept: str = None,
    doc_type: str = None,
):
    """
    Enrich → embed → store each chunk immediately after enriching.

    Rate limits applied independently:
      - Enrich : ENRICH_BATCH_SIZE  per minute, pause ENRICH_BATCH_WAIT s
      - Embed  : EMBED_BATCH_SIZE   per minute, pause EMBED_BATCH_WAIT  s

    Each chunk is written to the DB right after it's enriched and embedded,
    so no results are buffered in memory between stages.
    """
    total = len(indexed_texts)
    enrich_count = 0   # tracks requests in the current enrich window
    embed_count  = 0   # tracks requests in the current embed window

    for idx, (orig_idx, text) in enumerate(indexed_texts, start=1):

        # ── Enrich rate-limit gate ────────────────────────
        if enrich_count > 0 and enrich_count % ENRICH_BATCH_SIZE == 0:
            _wait(ENRICH_BATCH_WAIT, "enrich")

        try:
            metadata, enrich_usage = enrich(text)
            enrich_count += 1
            logger.info("[enrich %d/%d] ok — %s", idx, total, metadata["summary"][:60])
        except Exception as e:
            logger.error("[enrich %d/%d] failed — skipping chunk. Error: %s", idx, total, e)
            enrich_count += 1   # still counts against the rate limit
            continue            # skip store for this chunk

        # ── Embed rate-limit gate ─────────────────────────
        if embed_count > 0 and embed_count % EMBED_BATCH_SIZE == 0:
            _wait(EMBED_BATCH_WAIT, "embed")

        try:
            store(doc_id, orig_idx, text, metadata, enrich_usage, dept, doc_type)
            embed_count += 1
            logger.info("[store  %d/%d] saved to db", idx, total)
        except Exception as e:
            logger.error("[store  %d/%d] failed — %s", idx, total, e)
            embed_count += 1   # still counts against the rate limit


# ── Full pipeline ─────────────────────────────────────────
def ingest(file_path: str, dept: str = None, doc_type: str = None) -> str:
    """
    Orchestrates all four stages with rate-limit-aware batching:
      restructure → chunk → enrich (13/batch) → store/embed (98/batch)

    Enrich errors are skipped (chunk is dropped) without aborting the run.
    """
    db = get_db()
    logger.info("Ingesting: %s", file_path)

    # Register document
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (source_file, doc_type) VALUES (%s, %s) RETURNING id",
            (file_path, doc_type)
        )
        doc_id = cur.fetchone()["id"]
    db.commit()

    # Stage 1 — Restructure
    doc = restructure(file_path)
    logger.info("Parsed: %d text elements", len(doc.texts))

    # Stage 2 — Chunk
    chunks = chunk(doc)
    logger.info("Chunked into %d pieces", len(chunks))

    # Filter empty chunks early
    indexed_texts = [
        (i, ch.text.strip())
        for i, ch in enumerate(chunks)
        if ch.text.strip()
    ]

    # Stages 3 & 4 — Enrich then immediately store each chunk
    process_chunks(doc_id, indexed_texts, dept, doc_type)

    logger.info("Done. Doc ID: %s", doc_id)
    return doc_id


# ── Batch ingest from data folder ────────────────────────
def ingest_all(data_dir: str = "data", dept: str = None):
    """
    Ingest all supported documents from a directory one by one.
    Skips unsupported file types and logs any per-file failures.
    """
    supported = {".pdf", ".docx", ".html"}
    files = [
        f for f in Path(data_dir).iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not files:
        logger.warning("No supported files found in: %s", data_dir)
        return

    logger.info("Found %d file(s) in '%s'", len(files), data_dir)

    for file in files:
        try:
            ingest(str(file), dept=dept, doc_type=file.suffix.lower().lstrip("."))
        except Exception as e:
            logger.error("Failed to ingest %s — %s", file.name, e)


# ── Example usage ─────────────────────────────────────────
if __name__ == "__main__":
    ingest_all("data", "research")