from typing import List
from openai import OpenAI
from src.config import config
from src.utils.logger import logger

class EmbeddingService:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.model = model or config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embedding(self, text: str) -> List[float]:
        """Получить вектор для одиночной строки текста."""
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            clean_text = "empty"
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Получить векторы для списка строк пакетами."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() or "empty" for t in texts[i:i + batch_size]]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error generating batch embeddings (batch {i}..{i+len(batch)}): {e}")
                raise
        return all_embeddings
