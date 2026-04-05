import os
import threading
import logging
from google import genai
from google.genai import types
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Gemini singleton ──────────────────────────────────────
class GeminiClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    instance = super().__new__(cls)
                    instance.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                    instance.model = os.environ["GEMINI_GEN_AI_MODEL"] 
                    instance.embed_model = os.environ["GEMINI_EMBEDDING_MODEL"]  
                    cls._instance = instance
        return cls._instance

    def generate(self, prompt: str, json_mode: bool = False) -> tuple[str, dict]:
        """
        Generate text from a prompt.
        Returns (text, usage) where usage = {input_tokens, output_tokens, model}.
        Uses application/json response mode when json_mode=True (Gemini models only).
        """
        config = None
        if json_mode:
            config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            usage = {
                "input_tokens":  getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "model": self.model
            }
            return response.text, usage
        except Exception as e:
            logger.error("Gemini generate failed: %s", e)
            raise

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> tuple[list[float], dict]:
        """
        Embed a single string using gemini-embedding-001 (1536 dims).
        Returns (vector, usage) where usage = {input_tokens, output_tokens, model}.
        task_type should be RETRIEVAL_DOCUMENT for ingestion, RETRIEVAL_QUERY for search.
        """
        try:
            result = self.client.models.embed_content(
                model=self.embed_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=1536
                )
            )
            assert len(result.embeddings) == 1, (
                f"Expected 1 embedding, got {len(result.embeddings)}"
            )
            values = [float(v) for v in result.embeddings[0].values]
            usage = {
                "input_tokens":  getattr(result, "metadata", None) and getattr(result.metadata, "billable_character_count", 0) or 0,
                "output_tokens": 0,
                "model": self.embed_model
            }
            return values, usage
        except Exception as e:
            logger.error("Gemini embed failed: %s", e)
            raise


# ── NeonDB singleton ──────────────────────────────────────
class DBClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    instance = super().__new__(cls)
                    instance._conn = cls._connect()
                    cls._instance = instance
        return cls._instance

    @staticmethod
    def _connect() -> psycopg2.extensions.connection:
        """Open a fresh connection to NeonDB."""
        return psycopg2.connect(os.environ["NEON_DATABASE_URL"])

    @property
    def conn(self) -> psycopg2.extensions.connection:
        """
        Return a live connection.
        Neon serverless suspends after ~5 min of inactivity, which drops
        the TCP connection without setting conn.closed. A lightweight
        SELECT 1 probe catches this before any real query is attempted.
        """
        try:
            with self._conn.cursor() as probe:
                probe.execute("SELECT 1")
        except Exception:
            logger.warning("NeonDB connection lost — reconnecting.")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._connect()
        return self._conn

    def cursor(self) -> RealDictCursor:
        """
        Return a RealDictCursor so all query results come back as dicts.
        Explicitly passed here rather than on connect() for reliability
        across psycopg2 versions.
        """
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        """
        Close the connection and reset the singleton.
        Note: any code still holding a reference to this instance will
        have a stale object — call close() only at process shutdown.
        """
        with self._lock:
            try:
                if not self._conn.closed:
                    self._conn.close()
            except Exception as e:
                logger.warning("Error closing DB connection: %s", e)
            finally:
                DBClient._instance = None


# ── Convenience accessors ─────────────────────────────────
def get_gemini() -> GeminiClient:
    return GeminiClient()

def get_db() -> DBClient:
    return DBClient()