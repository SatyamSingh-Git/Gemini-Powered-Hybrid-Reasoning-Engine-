# src/reasoning_engine/core/retrieval.py (MODIFIED FOR GEMINI EMBEDDINGS)

import asyncio
import google.generativeai as genai
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Coroutine
from .models import DocumentChunk
import logging
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

# The new model identifier for Gemini embeddings
GEMINI_EMBEDDING_MODEL = "models/embedding-001"


class HybridRetriever:
    """
    A retriever that uses the Gemini API for dense embeddings and BM25 for sparse search.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        if not chunks:
            raise ValueError("Cannot initialize HybridRetriever with empty chunks.")

        self.chunks = chunks
        self.chunk_map: Dict[str, DocumentChunk] = {chunk.chunk_id: chunk for chunk in chunks}

    async def _embed_with_retry(self, content: str) -> List[float]:
        """A single embedding call with a simple retry mechanism."""
        try:
            result = await genai.embed_content_async(
                model=GEMINI_EMBEDDING_MODEL,
                content=content,
                task_type="RETRIEVAL_DOCUMENT"  # Crucial for retrieval tasks
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"API embedding failed for content: '{content[:50]}...'. Error: {e}")
            await asyncio.sleep(1)  # Wait a second before a potential retry
            return []  # Return empty on failure

    async def _build_indices_async(self):
        """Asynchronously builds FAISS and BM25 indices by calling the Gemini API."""
        logger.info(f"Embedding {len(self.chunks)} chunks via Gemini API. This may take a while...")

        # --- Sparse Index (BM25) - This can be done first and is fast ---
        texts = [chunk.text for chunk in self.chunks]
        tokenized_corpus = [doc.split(" ") for doc in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_map = {i: self.chunks[i] for i in range(len(self.chunks))}

        # --- Dense Index (FAISS) - This involves many API calls ---
        # Use a semaphore to limit concurrent API calls to avoid rate limiting.
        # Google's default is often 60 requests/minute, so a limit of 15-20 is safe.
        semaphore = asyncio.Semaphore(20)

        async def embed_task(chunk):
            async with semaphore:
                return await self._embed_with_retry(chunk.text)

        embedding_coroutines: List[Coroutine] = [embed_task(chunk) for chunk in self.chunks]

        # Use tqdm to show progress for the many API calls
        embeddings = await tqdm_asyncio.gather(*embedding_coroutines, desc="Embedding Chunks")

        # Filter out any failed embeddings
        valid_embeddings = [emb for emb in embeddings if emb]
        if not valid_embeddings:
            raise RuntimeError("Failed to generate any embeddings from the API.")

        embedding_dim = len(valid_embeddings[0])
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)

        # We need to map the FAISS index back to our original chunks
        self.chunk_ids = np.array([i for i, emb in enumerate(embeddings) if emb])
        self.faiss_index.add(np.array(valid_embeddings).astype('float32'))

    @classmethod
    async def create(cls, chunks: List[DocumentChunk]):
        """Asynchronous factory to create and initialize an instance."""
        retriever = cls(chunks)
        await retriever._build_indices_async()
        return retriever

    def _search_dense(self, query: str, k: int) -> Dict[str, float]:
        """Performs a dense search using the Gemini API and FAISS."""
        # Note: Using the synchronous `embed_content` here as it's a single call.
        query_embedding_result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"  # Use the specific query type
        )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')

        _, indices = self.faiss_index.search(query_embedding, k)

        results = {}
        for idx in indices[0]:
            if idx != -1:
                original_chunk_index = self.chunk_ids[idx]
                chunk_id = self.chunks[original_chunk_index].chunk_id
                results[chunk_id] = 1.0 / (len(results) + 1)
        return results

    # The _search_sparse and _reciprocal_rank_fusion methods remain unchanged.
    def _search_sparse(self, query: str, k: int) -> Dict[str, float]:
        tokenized_query = query.split(" ")
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        return {self.bm25_corpus_map[idx].chunk_id: doc_scores[idx] for idx in top_k_indices}

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        fused_scores = {}
        dense_ranks = {doc_id: i + 1 for i, doc_id in enumerate(dense_results.keys())}
        sparse_ranks = {doc_id: i + 1 for i, doc_id in enumerate(sparse_results.keys())}
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        for doc_id in all_doc_ids:
            score = 0
            if dense_rank := dense_ranks.get(doc_id): score += 1 / (k + dense_rank)
            if sparse_rank := sparse_ranks.get(doc_id): score += 1 / (k + sparse_rank)
            fused_scores[doc_id] = score
        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        logger.info(f"Retrieving chunks for query: '{query}'")
        dense_results = self._search_dense(query, k=top_k * 2)
        sparse_results = self._search_sparse(query, k=top_k * 2)
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        top_chunk_ids = [doc_id for doc_id, _ in fused_results[:top_k]]
        retrieved_chunks = [self.chunk_map[chunk_id] for chunk_id in top_chunk_ids]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks after fusion.")
        return retrieved_chunks