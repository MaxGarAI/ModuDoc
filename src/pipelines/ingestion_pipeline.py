from pathlib import Path
from typing import Any, Callable, Dict, Optional
from src.components.document_parser import DocumentParser
from src.components.chunker import DoclingChunker
from src.components.embedding_service import EmbeddingService
from src.components.vector_store import PineconeVectorStore
from src.components.llm_service import LLMService
from src.utils.logger import logger

class DocumentIngestionPipeline:
    """
    Пайплайн приема и обработки документов:
    1. Парсинг через Docling
    2. Чанкинг через Docling HybridChunker
    3. Векторизация чанков
    4. Сохранение в Pinecone (индекс pdf-bot)
    5. Автоматическая генерация структурированного резюме
    """
    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[DoclingChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[PineconeVectorStore] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DoclingChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or PineconeVectorStore()
        self.llm_service = llm_service or LLMService()

    async def run(
        self,
        file_path: Path | str,
        document_name: str,
        user_id: Optional[str | int] = None,
        status_callback: Optional[Callable[[str], Any]] = None
    ) -> Dict[str, Any]:
        path = Path(file_path)

        async def notify(status_text: str):
            if status_callback:
                try:
                    res = status_callback(status_text)
                    if hasattr(res, "__await__"):
                        await res
                except Exception as e:
                    logger.warning(f"Status callback failed: {e}")

        # 1. Парсинг документа Docling
        await notify("⚙️ Анализирую структуру документа через Docling...")
        parse_result = self.parser.parse(path)
        dl_doc = parse_result["dl_doc"]
        markdown_content = parse_result["markdown"]
        pages_count = parse_result["pages_count"]

        # 2. Чанкинг
        await notify("✂️ Разбиваю документ на смысловые фрагменты (чанки)...")
        chunks = self.chunker.chunk_document(
            dl_doc=dl_doc,
            document_name=document_name,
            user_id=user_id
        )

        if not chunks:
            raise ValueError(f"Не удалось извлечь текстовые фрагменты из документа {document_name}")

        # 3. Векторизация
        await notify(f"🧮 Генерирую эмбеддинги для {len(chunks)} фрагментов...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.get_embeddings_batch(texts)

        # 4. Сохранение в Pinecone
        await notify("💾 Сохраняю векторные данные в Pinecone...")
        pinecone_vectors = []
        for chunk_item, emb in zip(chunks, embeddings):
            pinecone_vectors.append({
                "id": chunk_item["id"],
                "values": emb,
                "metadata": chunk_item["metadata"]
            })

        self.vector_store.upsert_vectors(pinecone_vectors)

        # 5. Автоматическая генерация резюме
        await notify("📝 Составляю структурированное резюме документа...")
        summary = self.llm_service.generate_summary(
            document_name=document_name,
            document_text=markdown_content
        )

        logger.info(f"Successfully finished ingestion pipeline for '{document_name}'. Chunks: {len(chunks)}")
        return {
            "document_name": document_name,
            "chunks_count": len(chunks),
            "pages_count": pages_count,
            "summary": summary
        }
