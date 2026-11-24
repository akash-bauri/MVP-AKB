
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.services.rag.pipeline import ingest_all_pdfs_from_folder
from app.services import redis_cache
from app.database import chromadb_client


logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "banking_knowledge"


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("===================================================")
    logger.info("Starting 40 AKB Private Limited AI Backend...")

    # STEP 1: REDIS INITIALIZATION

    logger.info("Initializing Redis...")

    try:
        redis_cache.connect()

        redis_status = redis_cache.health_check()

        if redis_status.get("redis_connected"):
            logger.info("Redis connected successfully.")
        else:
            logger.warning(
                "Redis unavailable. Continuing without caching."
            )

    except Exception:
        logger.exception("Redis initialization failed.")
        logger.warning("Continuing without Redis cache.")

    # STEP 2: CHROMADB INITIALIZATION

    logger.info("Initializing ChromaDB...")

    try:
        chromadb_client.initialize_database()

        logger.info("ChromaDB initialized successfully.")

    except Exception:
        logger.exception("ChromaDB initialization failed.")
        raise

    # STEP 3: KNOWLEDGE BASE INITIALIZATION

    logger.info("Checking Knowledge Base...")

    try:
        collection = chromadb_client.get_collection(
            KNOWLEDGE_COLLECTION
        )

        if collection.count() == 0:

            logger.info(
                "Knowledge Base empty. Starting PDF indexing."
            )

            try:
                indexing_status = ingest_all_pdfs_from_folder(
                    folder_path="pdfs",
                    language="English"
                )

                logger.info(
                    "Knowledge Base indexing result: %s",
                    indexing_status
                )

            except Exception:
                logger.exception("PDF indexing failed.")

        else:

            logger.info(
                "Knowledge Base loaded with %s records.",
                collection.count()
            )

    except Exception:
        logger.exception("Knowledge Base initialization failed.")
        raise

    logger.info("40 AKB AI startup completed.")
    logger.info("===================================================")

    try:
        yield

    finally:

        logger.info(
            "Shutting down 40 AKB Private Limited AI Backend..."
        )

        # STEP 4: CLOSE REDIS

        try:
            redis_cache.close()
            logger.info("Redis connection closed.")

        except Exception:
            logger.exception("Redis shutdown failed.")

        # STEP 5: CLOSE CHROMADB

        try:
            chromadb_client.close_database()
            logger.info("ChromaDB references released.")

        except Exception:
            logger.exception("ChromaDB shutdown failed.")

        logger.info("40 AKB AI shutdown completed.")
        logger.info("===================================================")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="40 AKB Private Limited - AI Backend",
    description=(
        "Multilingual AI-powered financial education "
        "and learning platform."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "company": "40 AKB Private Limited",
        "product": "40 AKB AI",
        "message": "40 AKB Private Limited AI Backend is running.",
        "version": "1.0.0",
        "status": "healthy"
    }
