
import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List

import chromadb
from huggingface_hub import InferenceClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


# =====================================================================
# CONFIGURATION
# =====================================================================

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db_storage")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_API_KEY")
    or ""
).strip()

COLLECTION_METADATA_BLUEPRINT = {
    "hnsw:space": "cosine",
    "project": "40 AKB AI",
    "system_version": "1.0.0",
    "description": "Multilingual financial knowledge base vector storage."
}

logger = logging.getLogger(__name__)


# =====================================================================
# REMOTE HUGGING FACE EMBEDDING FUNCTION
# =====================================================================

class ZeroRamHuggingFaceEmbedding(EmbeddingFunction):

    """
    Generates embeddings remotely through Hugging Face.
    No local Sentence Transformers or PyTorch model is loaded.
    """

    def __init__(self, api_key: str, model_name: str):

        if not api_key:
            raise RuntimeError(
                "Hugging Face token missing. Configure HF_TOKEN "
                "or HUGGINGFACE_API_KEY in your environment."
            )

        self.api_key = api_key
        self.model_name = model_name

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key
        )

        self.max_retries = 3
        self.backoff_delays = [1, 2, 4]

        logger.info(
            "Hugging Face remote embedding client initialized: %s",
            self.model_name
        )

    def __call__(self, input: Documents) -> Embeddings:

        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Embedding input contains empty or invalid text.")

        for attempt in range(1, self.max_retries + 1):

            try:
                logger.info(
                    "Requesting Hugging Face embeddings. "
                    "Batch size: %s, Attempt: %s",
                    len(texts),
                    attempt
                )

                result = self.client.feature_extraction(
                    text=texts,
                    model=self.model_name
                )

                if hasattr(result, "tolist"):
                    result = result.tolist()

                if not isinstance(result, list) or not result:
                    raise RuntimeError(
                        "Hugging Face returned an empty embedding response."
                    )

                # Ensure output has shape: list[list[float]]
                if isinstance(result[0], (int, float)):
                    result = [result]

                if len(result) != len(texts):
                    raise RuntimeError(
                        "Embedding count mismatch. "
                        f"Expected {len(texts)}, received {len(result)}."
                    )

                embeddings = []

                for vector in result:

                    if not isinstance(vector, (list, tuple)):
                        raise RuntimeError(
                            "Invalid embedding vector returned by Hugging Face."
                        )

                    converted_vector = [float(value) for value in vector]

                    if not converted_vector:
                        raise RuntimeError("Received an empty embedding vector.")

                    embeddings.append(converted_vector)

                # All vectors in one batch must have the same dimension.
                dimensions = {len(vector) for vector in embeddings}

                if len(dimensions) != 1:
                    raise RuntimeError(
                        "Inconsistent embedding dimensions returned by API."
                    )

                logger.info(
                    "Hugging Face embedding generation successful. "
                    "Dimension: %s",
                    len(embeddings[0])
                )

                return embeddings

            except Exception as error:

                logger.exception(
                    "Hugging Face embedding request failed on attempt %s: %s",
                    attempt,
                    str(error)
                )

                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Hugging Face embedding generation failed after "
                        f"{self.max_retries} attempts: {error}"
                    ) from error

                delay = self.backoff_delays[attempt - 1]

                logger.warning(
                    "Retrying Hugging Face request in %s seconds.",
                    delay
                )

                time.sleep(delay)

        raise RuntimeError("Embedding retry process exited unexpectedly.")


# =====================================================================
# DATABASE SINGLETONS
# =====================================================================

_init_lock = threading.RLock()

_initialized = False

_chroma_client: Optional[ClientAPI] = None

_embedding_function: Optional[ZeroRamHuggingFaceEmbedding] = None

_collection_cache: Dict[str, Collection] = {}


# =====================================================================
# DATABASE INITIALIZATION
# =====================================================================

def initialize_database() -> None:

    """
    Initializes persistent ChromaDB and remote Hugging Face embeddings.

    Must be called during FastAPI startup.
    """

    global _chroma_client
    global _embedding_function
    global _initialized

    if _initialized:
        return

    with _init_lock:

        if _initialized:
            return

        try:

            logger.info(
                "Initializing 40 AKB AI ChromaDB at: %s",
                CHROMA_DB_PATH
            )

            os.makedirs(CHROMA_DB_PATH, exist_ok=True)

            # Initialize persistent ChromaDB.
            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH
            )

            # Initialize remote embedding service.
            token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_API_KEY")
                or ""
            ).strip()

            _embedding_function = ZeroRamHuggingFaceEmbedding(
                api_key=token,
                model_name=EMBEDDING_MODEL_NAME
            )

            _initialized = True

            logger.info(
                "40 AKB AI ChromaDB and Hugging Face "
                "embedding service initialized successfully."
            )

        except Exception as error:

            _chroma_client = None
            _embedding_function = None
            _initialized = False

            logger.critical(
                "Database initialization failed: %s",
                str(error),
                exc_info=True
            )

            raise RuntimeError(
                f"ChromaDB initialization failure: {error}"
            ) from error


# =====================================================================
# COLLECTION MANAGEMENT
# =====================================================================

def get_collection(name: str) -> Collection:

    """
    Creates or retrieves a ChromaDB collection.

    All collections use the same remote Hugging Face
    embedding model for ingestion and retrieval.
    """

    if not _initialized:
        raise RuntimeError(
            "Database is not initialized. "
            "Call initialize_database() during application startup."
        )

    if not name or not name.strip():
        raise ValueError("Collection name cannot be empty.")

    if name in _collection_cache:
        return _collection_cache[name]

    with _init_lock:

        if name in _collection_cache:
            return _collection_cache[name]

        try:

            logger.info(
                "Loading ChromaDB collection: %s",
                name
            )

            metadata = COLLECTION_METADATA_BLUEPRINT.copy()

            metadata["collection_identifier_name"] = name

            collection = _chroma_client.get_or_create_collection(
                name=name,
                embedding_function=_embedding_function,
                metadata=metadata
            )

            _collection_cache[name] = collection

            logger.info(
                "Collection successfully initialized: %s",
                name
            )

            return collection

        except Exception as error:

            logger.error(
                "Collection initialization failed for %s: %s",
                name,
                str(error),
                exc_info=True
            )

            raise RuntimeError(
                f"Failed to initialize collection '{name}': {error}"
            ) from error


# =====================================================================
# DATABASE HEALTH CHECK
# =====================================================================

def check_database() -> Dict[str, Any]:

    """
    Checks ChromaDB connectivity and embedding configuration.
    """

    start_time = time.time()

    is_healthy = False

    diagnostic_message = "Database is not initialized."

    cached_partitions_count = 0

    try:

        if not _initialized:
            raise RuntimeError("Database is not initialized.")

        if _chroma_client is None:
            raise RuntimeError("ChromaDB client is unavailable.")

        if _embedding_function is None:
            raise RuntimeError("Embedding function is unavailable.")

        _chroma_client.heartbeat()

        cached_partitions_count = len(_collection_cache)

        is_healthy = True

        diagnostic_message = (
            "ChromaDB is reachable and remote embedding "
            "configuration is initialized."
        )

    except Exception as error:

        diagnostic_message = str(error)

        logger.error(
            "Database health check failed: %s",
            diagnostic_message
        )

    elapsed_ms = round(
        (time.time() - start_time) * 1000,
        2
    )

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database_reachable": is_healthy,
        "database_path": CHROMA_DB_PATH,
        "embedding_provider": "Hugging Face Inference API",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cached_collections_count": cached_partitions_count,
        "latency_ms": elapsed_ms,
        "message": diagnostic_message
    }


# =====================================================================
# DATABASE SHUTDOWN
# =====================================================================

def close_database() -> None:

    """
    Clears local singleton references during application shutdown.
    """

    global _chroma_client
    global _embedding_function
    global _initialized

    with _init_lock:

        _collection_cache.clear()

        _embedding_function = None
        _chroma_client = None
        _initialized = False

        logger.info("ChromaDB local references released.")
