from typing import Any, Dict, List, Optional
from pinecone import Pinecone
from src.config import config
from src.utils.logger import logger

class PineconeVectorStore:
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or config.PINECONE_KEY
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        vectors: список словарей вида:
        [
            {
                "id": "doc123_chunk_0",
                "values": [0.1, 0.2, ...],
                "metadata": {
                    "text": "...",
                    "document_name": "...",
                    "user_id": "...",
                    "chunk_index": 0,
                    "page_number": 1,
                    "headings": "..."
                }
            }
        ]
        """
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
            except Exception as e:
                logger.error(f"Error upserting vectors batch {i}..{i+len(batch)}: {e}")
                raise
        logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone index '{self.index_name}'")
        return total_upserted

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        user_id: Optional[str | int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Выполняет поиск ближайших векторов.
        Возвращает список результатов с текстом и метаданными.
        """
        search_filter = {}
        if user_id is not None:
            search_filter["user_id"] = str(user_id)
        if filter_dict:
            search_filter.update(filter_dict)

        try:
            kwargs: Dict[str, Any] = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": True
            }
            if search_filter:
                kwargs["filter"] = search_filter

            response = self.index.query(**kwargs)
            matches = response.get("matches", [])

            results = []
            for match in matches:
                score = match.get("score", 0.0)
                if score >= min_score:
                    results.append({
                        "id": match.get("id"),
                        "score": score,
                        "metadata": match.get("metadata", {}),
                        "text": match.get("metadata", {}).get("text", "")
                    })
            return results
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            raise

    def delete_by_user(self, user_id: str | int) -> None:
        """Удаляет все векторы конкретного пользователя."""
        try:
            self.index.delete(filter={"user_id": str(user_id)})
            logger.info(f"Deleted vectors for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Error deleting vectors for user {user_id}: {e}")
            raise

    def delete_by_document(self, document_name: str, user_id: Optional[str | int] = None) -> None:
        """Удаляет векторы конкретного документа."""
        try:
            delete_filter = {"document_name": document_name}
            if user_id is not None:
                delete_filter["user_id"] = str(user_id)
            self.index.delete(filter=delete_filter)
            logger.info(f"Deleted vectors for document: {document_name}")
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_name}: {e}")
            raise
