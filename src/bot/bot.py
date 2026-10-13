from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)
from src.config import config
from src.bot.handlers import (
    start_handler,
    help_handler,
    clear_handler,
    status_handler,
    document_handler,
    message_handler,
)
from src.utils.logger import logger

def create_bot_application() -> Application:
    """Создает и настраивает экземпляр Telegram-приложения."""
    config.validate()

    logger.info("Initializing Telegram bot application...")
    application = Application.builder().token(config.TELEGRAM_KEY).build()

    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("clear", clear_handler))
    application.add_handler(CommandHandler("status", status_handler))

    # Регистрация обработчика входящих документов
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    # Регистрация обработчика текстовых сообщений (вопросов)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    return application

def start_polling() -> None:
    """Запускает бота в режиме long polling."""
    app = create_bot_application()
    logger.info("Starting Telegram bot polling...")
    app.run_polling(drop_pending_updates=True)
