import asyncio
from pathlib import Path
import sys

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components.document_parser import DocumentParser
from src.components.chunker import DoclingChunker
from src.components.embedding_service import EmbeddingService
from src.components.vector_store import PineconeVectorStore
from src.components.llm_service import LLMService
from src.pipelines.ingestion_pipeline import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGQueryPipeline
from src.utils.logger import logger

async def test_components_and_pipeline():
    logger.info("=== 1. ТЕСТ: Создание тестового документа ===")
    test_dir = Path(__file__).resolve().parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "quantum_computing.md"
    test_file.write_text(
        """# Введение в квантовые вычисления

Квантовые вычисления — это область информатики, использующая принципы квантовой механики, такие как суперпозиция и квантовая запутанность, для решения вычислительных задач.

## Архитектура кубитов
В отличие от классических битов, принимающих значения 0 или 1, квантовый бит (кубит) может находиться в суперпозиции обоих состояний. Основные реализации кубитов включают:
- Сверхпроводящие контуры (используются IBM и Google)
- Ионные ловушки (IonQ)
- Фотонные чипы

## Квантовое превосходство
Квантовое превосходство означает демонстрацию того, что программируемое квантовое устройство способно решить задачу, практически невыполнимую на классическом суперкомпьютере за разумное время.

## Алгоритмы
Наиболее известные квантовые алгоритмы:
1. Алгоритм Шора — факторизация целых чисел за полиномиальное время.
2. Алгоритм Гровера — квадратичное ускорение поиска в неструктурированных базах данных.
""",
        encoding="utf-8"
    )

    test_user_id = 999999

    logger.info("=== 2. ТЕСТ: DocumentIngestionPipeline ===")
    pipeline = DocumentIngestionPipeline()
    ingest_result = await pipeline.run(
        file_path=test_file,
        document_name="quantum_computing.md",
        user_id=test_user_id,
        status_callback=lambda msg: logger.info(f"Progress: {msg}")
    )

    logger.info(f"Ingestion result:\nDoc: {ingest_result['document_name']}\nChunks: {ingest_result['chunks_count']}\nSummary:\n{ingest_result['summary']}")

    assert ingest_result["chunks_count"] > 0, "No chunks produced!"
    assert len(ingest_result["summary"]) > 50, "Summary too short!"

    logger.info("=== 3. ТЕСТ: RAGQueryPipeline ===")
    rag = RAGQueryPipeline()
    rag_result = rag.run(
        question="Какое преимущество дает алгоритм Шора?",
        user_id=test_user_id
    )

    logger.info(f"RAG Answer:\n{rag_result['answer']}")
    logger.info(f"Found chunks: {rag_result['chunks_found']}")
    logger.info(f"Sources: {rag_result['sources']}")

    assert rag_result["chunks_found"] > 0, "No chunks found in Pinecone!"
    assert "Шор" in rag_result["answer"] or "факторизац" in rag_result["answer"], "Answer does not mention Shor algorithm!"

    logger.info("=== 4. ТЕСТ: Очистка тестовых данных из Pinecone ===")
    vs = PineconeVectorStore()
    vs.delete_by_user(test_user_id)
    logger.info("Test vectors deleted successfully.")

    logger.info("🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")

if __name__ == "__main__":
    asyncio.run(test_components_and_pipeline())
