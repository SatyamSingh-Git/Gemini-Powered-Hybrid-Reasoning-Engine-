import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from ..core.models import APIRequest, APIResponse, Answer
from ..core.ingestion import process_documents
from ..core.retrievel import HybridRetriever
from ..core.agent import ReasoningAgent

logger = logging.getLogger(__name__)

router = APIRouter()
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# A simple hardcoded bearer token for the hackathon
EXPECTED_BEARER_TOKEN = "Bearer bc916bc507a9b3b680e613c91243b99771a30be1587ca8d9eb8cc4b9dfab5a55"


async def verify_token(authorization: str = Security(api_key_header)):
    """Dependency to verify the bearer token."""
    if authorization != EXPECTED_BEARER_TOKEN:
        logger.warning(f"Invalid authentication token received: {authorization}")
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token",
        )


@router.post(
    "/hackrx/run",
    response_model=APIResponse,
    tags=["Reasoning Engine"],
    dependencies=[Depends(verify_token)],
)
async def run_reasoning_engine(request: APIRequest) -> APIResponse:
    """
    The main endpoint to process documents and answer questions using Gemini embeddings.
    Orchestrates the entire workflow:
    1.  Ingests and chunks documents from URLs.
    2.  Asynchronously builds a hybrid search index using the Gemini Embedding API.
    3.  Initializes a reasoning agent.
    4.  Processes each question concurrently to generate answers.
    5.  Returns a structured, explainable response.
    """
    request_start_time = time.time()
    logger.info("================== NEW REQUEST RECEIVED ==================")
    logger.info(f"Received {len(request.documents)} docs and {len(request.questions)} questions.")

    try:
        # --- CHECKPOINT 1: DOCUMENT INGESTION ---
        logger.info("[1/5] Starting document ingestion...")
        ingestion_start_time = time.time()
        all_chunks = await process_documents([str(url) for url in request.documents])
        logger.info(
            f"[1/5] Document ingestion complete. Found {len(all_chunks)} chunks. (Took {time.time() - ingestion_start_time:.2f}s)")
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Failed to process any of the provided documents.")

        # --- CHECKPOINT 2: RETRIEVER INITIALIZATION (NOW ASYNC) ---
        logger.info("[2/5] Initializing API-based Hybrid Retriever...")
        retriever_start_time = time.time()
        # This is the key change: calling the async factory method
        retriever = await HybridRetriever.create(chunks=all_chunks)
        logger.info(
            f"[2/5] Hybrid Retriever initialized successfully. (Took {time.time() - retriever_start_time:.2f}s)")

        # --- CHECKPOINT 3: AGENT INITIALIZATION ---
        logger.info("[3/5] Initializing Reasoning Agent...")
        agent = ReasoningAgent(retriever=retriever)
        logger.info("[3/5] Reasoning Agent initialized.")

        # --- CHECKPOINT 4: CONCURRENT REASONING ---
        logger.info("[4/5] Starting to process all questions concurrently...")
        reasoning_start_time = time.time()
        answer_tasks = [agent.answer_question(query) for query in request.questions]
        structured_answers: list[Answer] = await asyncio.gather(*answer_tasks)
        logger.info(f"[4/5] All questions processed. (Took {time.time() - reasoning_start_time:.2f}s)")

        # --- CHECKPOINT 5: FORMATTING RESPONSE ---
        logger.info("[5/5] Formatting final response...")
        simple_answers = [ans.simple_answer for ans in structured_answers]

        total_time = time.time() - request_start_time
        logger.info(f"================== REQUEST COMPLETE (Total Time: {total_time:.2f}s) ==================")
        return APIResponse(answers=simple_answers)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")