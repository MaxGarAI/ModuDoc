import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import logger
from src.bot.bot import start_polling

def main():
    try:
        logger.info("Starting Docling RAG Bot...")
        start_polling()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error while running bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()
