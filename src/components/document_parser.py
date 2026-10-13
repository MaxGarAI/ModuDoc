from pathlib import Path
from typing import Any, Dict, Optional
from src.utils.logger import logger

class DocumentParser:
    """
    Компонент для парсинга документов различных форматов (PDF, DOCX, PPTX, HTML, TXT)
    с использованием Docling DocumentConverter.
    """
    def __init__(self):
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
                logger.info("Docling DocumentConverter successfully initialized.")
            except ImportError as e:
                logger.error(f"Docling is not installed or failed to import: {e}")
                raise
        return self._converter

    def parse(self, file_path: Path | str) -> Dict[str, Any]:
        """
        Конвертирует документ и возвращает структурированное представление:
        - dl_doc: исходный объект DoclingDocument
        - markdown: полный текст документа в markdown-формате
        - filename: исходное имя файла
        - num_pages: количество страниц (если применимо)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Parsing document with Docling: {path.name}")
        try:
            conversion_result = self.converter.convert(path)
            dl_doc = conversion_result.document
            markdown_content = dl_doc.export_to_markdown()

            num_pages = getattr(dl_doc, "pages", None)
            pages_count = len(num_pages) if num_pages is not None else 1

            logger.info(f"Document {path.name} parsed successfully. Pages: {pages_count}, Length: {len(markdown_content)} chars")
            return {
                "dl_doc": dl_doc,
                "markdown": markdown_content,
                "filename": path.name,
                "pages_count": pages_count
            }
        except Exception as e:
            logger.error(f"Error parsing document {path.name} with Docling: {e}")
            raise
