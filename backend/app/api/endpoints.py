
import os
import time
import uuid
import logging

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
)

from starlette.concurrency import run_in_threadpool

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import ask_the_principal, MODEL_NAME
from app.services import whisper_service, redis_cache
from app.services.rag.pipeline import (
    search_bookshelf,
    ingest_all_pdfs_from_folder,
)
from app.database import chromadb_client


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_health_check(check_function):
    try:
        return check_function()
    except Exception:
        logger.exception("Health check failed")
        return {"status": "unavailable"}


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize supported language names and codes."""
    if not language:
        return None

    value = language.strip().lower()

    aliases = {
        "english": "en",
        "en-in": "en",
        "hindi": "hi",
        "hi-in": "hi",
        "bengali": "bn",
        "bangla": "bn",
        "bn-in": "bn",
    }

    value = aliases.get(value, value)

    if value not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Supported languages are English (en), Hindi (hi), and Bengali (bn).",
        )

    return value


def _normalize_detected_language(language: Optional[str]) -> Optional[str]:
    """Convert Whisper language names into supported language codes."""
    if not language:
        return None

    value = language.strip().lower()

    aliases = {
        "english": "en",
        "hindi": "hi",
        "bengali": "bn",
        "bangla": "bn",
        "en-in": "en",
        "hi-in": "hi",
        "bn-in": "bn",
    }

    value = aliases.get(value, value)

    return value if value in SUPPORTED_LANGUAGES else None


async def process_core_chat_pipeline(
    question: str,
    request_id: str,
    start_time: float,
    language: Optional[str] = None,
    detected_language: Optional[str] = None,
) -> ChatResponse:

    normalized_question = (question or "").strip()

    if not normalized_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a question.",
        )

    if len(normalized_question) > 5000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Question exceeds the 5000-character limit.",
        )

    # Explicit selection takes priority.
    # Otherwise, use Whisper's detected language when available.
    response_language = (
        _normalize_language(language)
        or _normalize_language(detected_language)
    )

    cache_key = None

    # 1. Generate language-aware cache key.
    try:
        cache_input = (
            f"{MODEL_NAME}:"
            f"{response_language or 'auto'}:"
            f"{normalized_question}"
        )

        cache_key = await run_in_threadpool(
            redis_cache.generate_cache_key,
            cache_input,
        )

    except Exception:
        logger.warning(
            "[%s] Cache-key generation failed.",
            request_id,
            exc_info=True,
        )

    # 2. Redis cache lookup.
    if cache_key:
        try:
            cached_response = await run_in_threadpool(
                redis_cache.get,
                cache_key,
            )

            if (
                isinstance(cached_response, dict)
                and cached_response.get("answer")
                and cached_response.get("success", True)
            ):
                elapsed = round(time.time() - start_time, 3)

                return ChatResponse(
                    success=True,
                    question=normalized_question,
                    answer=cached_response["answer"],
                    detected_language=cached_response.get(
                        "detected_language", "Unknown"
                    ),
                    source="Redis Cache",
                    response_time=elapsed,
                    timestamp=_timestamp(),
                )

        except Exception:
            logger.warning(
                "[%s] Cache read failed.",
                request_id,
                exc_info=True,
            )

    # 3. Search the existing PDF knowledge base.
    context = ""
    source_identity = "General AI Knowledge"

    try:
        rag_result = await run_in_threadpool(
            search_bookshelf,
            normalized_question,
        )

        if rag_result and rag_result.get("found"):
            context = rag_result.get("context", "") or ""
            source_identity = rag_result.get(
                "source",
                "PDF Knowledge Base",
            )

    except Exception:
        logger.exception(
            "[%s] RAG search failed.",
            request_id,
        )

    # 4. Generate multilingual AI answer.
    try:
        ai_output = await run_in_threadpool(
            ask_the_principal,
            normalized_question,
            context,
            response_language,
        )

        if not isinstance(ai_output, dict):
            raise ValueError("Invalid LLM response structure.")

        final_answer = str(
            ai_output.get("answer") or ""
        ).strip()

        detected = ai_output.get(
            "detected_language",
            "Unknown",
        )

        llm_success = ai_output.get("success", True) is True

        if not llm_success or not final_answer:
            raise ValueError("LLM did not generate a valid answer.")

    except Exception:
        logger.exception(
            "[%s] LLM generation failed.",
            request_id,
        )

        return ChatResponse(
            success=False,
            question=normalized_question,
            answer=(
                "The AI service is temporarily unavailable. "
                "Please try again shortly."
            ),
            detected_language=(
                SUPPORTED_LANGUAGES.get(response_language, "Unknown")
            ),
            source="Groq",
            response_time=round(time.time() - start_time, 3),
            timestamp=_timestamp(),
        )

    # 5. Cache successful answers.
    if cache_key and final_answer:
        try:
            cache_payload: Dict[str, Any] = {
                "answer": final_answer,
                "detected_language": detected,
                "success": True,
                "model": MODEL_NAME,
                "response_language": response_language or "auto",
            }

            await run_in_threadpool(
                redis_cache.set,
                cache_key,
                cache_payload,
            )

        except Exception:
            logger.warning(
                "[%s] Cache write failed.",
                request_id,
                exc_info=True,
            )

    elapsed = round(time.time() - start_time, 3)

    logger.info(
        "[%s] Chat completed | model=%s | language=%s | source=%s | latency=%.3fs",
        request_id,
        MODEL_NAME,
        response_language or detected,
        source_identity,
        elapsed,
    )

    return ChatResponse(
        success=True,
        question=normalized_question,
        answer=final_answer,
        detected_language=detected,
        source=source_identity,
        response_time=elapsed,
        timestamp=_timestamp(),
    )


# ============================================================
# TEXT CHAT
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def text_chat_endpoint(payload: ChatRequest):

    start_time = time.time()
    request_id = str(uuid.uuid4())

    question = (payload.question or "").strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    language = _normalize_language(payload.language)

    return await process_core_chat_pipeline(
        question=question,
        request_id=request_id,
        start_time=start_time,
        language=language,
    )


# ============================================================
# VOICE CHAT
# ============================================================

@router.post("/voice-chat", response_model=ChatResponse)
async def voice_chat_endpoint(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):

    start_time = time.time()
    request_id = str(uuid.uuid4())

    if not audio_file or not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a valid audio file.",
        )

    selected_language = _normalize_language(language)

    try:
        audio_bytes = await audio_file.read(
            MAX_AUDIO_BYTES + 1
        )

        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty.",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file exceeds the 25 MB limit.",
            )

        from io import BytesIO

        audio_stream = BytesIO(audio_bytes)

        result = await run_in_threadpool(
            whisper_service.transcribe_audio,
            audio_stream,
            audio_file.filename,
            audio_file.content_type,
        )

        if not result or not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Audio transcription failed. Please try again.",
            )

        question = str(
            result.get("transcription") or ""
        ).strip()

        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech was detected.",
            )

        whisper_language = _normalize_detected_language(
            result.get("detected_language")
        )

        response = await process_core_chat_pipeline(
            question=question,
            request_id=request_id,
            start_time=start_time,
            language=selected_language or whisper_language,
            detected_language=whisper_language,
        )

        response.response_time = round(
            time.time() - start_time,
            3,
        )

        return response

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "[%s] Voice processing failed.",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Voice processing is temporarily unavailable.",
        )

    finally:
        await audio_file.close()


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():

    redis_status = _safe_health_check(
        redis_cache.health_check
    )

    chroma_status = _safe_health_check(
        chromadb_client.check_database
    )

    whisper_status = _safe_health_check(
        whisper_service.health_check
    )

    return {
        "api": "healthy",
        "redis": redis_status,
        "chromadb": chroma_status,
        "whisper": whisper_status,
        "llm": {
            "provider": "Groq",
            "model": MODEL_NAME,
            "status": "configured",
        },
        "supported_languages": ["en", "hi", "bn"],
        "timestamp": _timestamp(),
    }


# ============================================================
# PDF REINDEX
# ============================================================

@router.post("/reindex", status_code=status.HTTP_200_OK)
def trigger_reindexing():

    start_time = time.time()

    try:
        indexing_summary = ingest_all_pdfs_from_folder(
            PDF_FOLDER
        )

        elapsed = round(time.time() - start_time, 3)

        logger.info(
            "PDF reindex completed | latency=%.3fs",
            elapsed,
        )

        return {
            "success": True,
            "indexed_documents": indexing_summary,
            "execution_time": f"{elapsed}s",
            "message": "PDF knowledge-base indexing completed.",
        }

    except Exception:
        logger.exception("PDF reindex failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF indexing failed. Check server logs.",
        )
