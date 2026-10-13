import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # Project paths
    BASE_DIR: Path = BASE_DIR
    TEMP_DIR: Path = BASE_DIR / "temp_files"

    # API Keys & Endpoints
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    TELEGRAM_KEY: str = os.getenv("TELEGRAM_KEY", "")
    PINECONE_KEY: str = os.getenv("PINECONE_KEY", "")

    # Pinecone configuration
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "pdf-bot")
    EMBEDDING_DIMENSION: int = 1536

    # Model configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    
    # RAG parameters
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25  # Cosine similarity threshold for relevance
    MAX_TOKENS_PER_CHUNK: int = 800

    @classmethod
    def validate(cls) -> None:
        missing = []
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.TELEGRAM_KEY:
            missing.append("TELEGRAM_KEY")
        if not cls.PINECONE_KEY:
            missing.append("PINECONE_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")


# Ensure temp directory exists
Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
config = Config()
