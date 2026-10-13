from typing import Any, Dict, List, Optional
from src.components.embedding_service import EmbeddingService
from src.components.vector_store import PineconeVectorStore
from src.components.llm_service import LLMService
from src.config import config
from src.utils.logger import logger

class RAGQueryPipeline:
    """
    Пайплайн ответов на вопросы по документам:
    1. Эмбеддинг вопроса пользователя
    2. Поиск релевантных чанков в Pinecone (с учетом user_id и порога сходства)
    3. Синтез ответа с помощью LLM с цитированием первоисточников
    """
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[PineconeVectorStore] = None,
        llm_service: Optional[LLMService] = None,
        top_k: int = None,
        similarity_threshold: float = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or PineconeVectorStore()
        self.llm_service = llm_service or LLMService()
        self.top_k = top_k or config.DEFAULT_TOP_K
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD

    def run(
        self,
        question: str,
        user_id: Optional[str | int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        logger.info(f"RAG query received from user {user_id}: '{question}'")

        # 1. Получаем эмбеддинг вопроса
        query_vector = self.embedding_service.get_embedding(question)

        # 2. Ищем релевантные чанки в Pinecone
        matches = self.vector_store.query(
            vector=query_vector,
            top_k=self.top_k,
            user_id=user_id,
            min_score=self.similarity_threshold
        )

        logger.info(f"Found {len(matches)} relevant chunks above threshold {self.similarity_threshold}")

        # 3. Синтез ответа через LLM
        answer = self.llm_service.answer_question(
            question=question,
            context_chunks=matches,
            chat_history=chat_history
        )

        # 4. Формируем список источников для метаданных
        sources = []
        for match in matches:
            meta = match.get("metadata", {})
            sources.append({
                "document": meta.get("document_name"),
                "page": meta.get("page_number"),
                "headings": meta.get("headings"),
                "score": round(match.get("score", 0.0), 3)
            })

        return {
            "answer": answer,
            "sources": sources,
            "chunks_found": len(matches)
        }
