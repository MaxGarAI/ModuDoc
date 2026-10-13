import hashlib
from typing import Any, Dict, List, Optional
from src.config import config
from src.utils.logger import logger

class DoclingChunker:
    """
    Компонент для разбиения структурированного документа на чанки
    с использованием Docling HybridChunker.
    """
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or config.MAX_TOKENS_PER_CHUNK
        self._chunker = None

    @property
    def chunker(self):
        if self._chunker is None:
            try:
                from docling.chunking import HybridChunker
                # Инициализация HybridChunker
                self._chunker = HybridChunker(max_tokens=self.max_tokens)
                logger.info(f"Docling HybridChunker initialized with max_tokens={self.max_tokens}")
            except Exception as e:
                logger.warning(f"Could not initialize HybridChunker with max_tokens: {e}. Trying default HybridChunker...")
                try:
                    from docling.chunking import HybridChunker
                    self._chunker = HybridChunker()
                except Exception as inner_e:
                    logger.error(f"Failed to initialize Docling HybridChunker: {inner_e}")
                    raise
        return self._chunker

    def chunk_document(
        self,
        dl_doc: Any,
        document_name: str,
        user_id: Optional[str | int] = None
    ) -> List[Dict[str, Any]]:
        """
        Разбивает DoclingDocument на список словарей с текстом и метаданными.
        """
        doc_hash = hashlib.md5(f"{document_name}_{user_id}".encode("utf-8")).hexdigest()[:10]
        chunks_data: List[Dict[str, Any]] = []

        try:
            raw_chunks = list(self.chunker.chunk(dl_doc=dl_doc))
            total_chunks = len(raw_chunks)
            logger.info(f"Generated {total_chunks} chunks for '{document_name}' using Docling HybridChunker.")

            for idx, raw_chunk in enumerate(raw_chunks):
                # Извлекаем текст
                text = getattr(raw_chunk, "text", str(raw_chunk)).strip()
                if not text:
                    continue

                # Извлекаем метаданные: заголовки и страницы
                headings_str = ""
                page_no = 1
                meta = getattr(raw_chunk, "meta", None)
                if meta:
                    headings = getattr(meta, "headings", [])
                    if headings:
                        headings_str = " > ".join(str(h) for h in headings)

                    doc_items = getattr(meta, "doc_items", [])
                    if doc_items and hasattr(doc_items[0], "prov") and doc_items[0].prov:
                        page_no = getattr(doc_items[0].prov[0], "page_no", 1)

                chunk_id = f"{doc_hash}_{idx}"
                metadata = {
                    "text": text,
                    "document_name": document_name,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "headings": headings_str,
                    "page_number": int(page_no)
                }
                if user_id is not None:
                    metadata["user_id"] = str(user_id)

                chunks_data.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata
                })

        except Exception as e:
            logger.error(f"Error during Docling chunking for {document_name}: {e}. Fallback to text chunking.")
            # Запасной вариант разбиения по абзацам в случае ошибки в гибридном чанкере
            markdown_text = getattr(dl_doc, "export_to_markdown", lambda: str(dl_doc))()
            chunks_data = self._fallback_text_chunking(markdown_text, document_name, user_id, doc_hash)

        return chunks_data

    def _fallback_text_chunking(
        self,
        text: str,
        document_name: str,
        user_id: Optional[str | int],
        doc_hash: str
    ) -> List[Dict[str, Any]]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            if current_len + len(p_strip) > 1500 and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p_strip]
                current_len = len(p_strip)
            else:
                current_chunk.append(p_strip)
                current_len += len(p_strip)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        result = []
        total = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_hash}_{idx}"
            meta = {
                "text": chunk_text,
                "document_name": document_name,
                "chunk_index": idx,
                "total_chunks": total,
                "headings": "",
                "page_number": 1
            }
            if user_id is not None:
                meta["user_id"] = str(user_id)
            result.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": meta
            })
        return result
