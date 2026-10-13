import html
from typing import Dict, List
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes
from src.config import config
from src.pipelines.ingestion_pipeline import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGQueryPipeline
from src.components.vector_store import PineconeVectorStore
from src.utils.file_storage import FileStorage
from src.utils.logger import logger

# Инициализируем пайплайны
ingestion_pipeline = DocumentIngestionPipeline()
rag_pipeline = RAGQueryPipeline()
vector_store = PineconeVectorStore()

# Память короткого контекста диалога (user_id -> list of {"role": "...", "content": "..."})
user_chat_histories: Dict[int, List[Dict[str, str]]] = {}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".html"}

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_text = (
        f"👋 Здравствуйте, {user.first_name if user else 'друг'}!\n\n"
        "Я интеллектуальный **RAG-ассистент** для анализа документов на базе **Docling** и **Pinecone**.\n\n"
        "✨ **Что я умею:**\n"
        "1. 📄 **Анализ файлов**: отправьте мне файл (PDF, DOCX, PPTX, TXT, MD), я обработаю его с помощью Docling.\n"
        "2. 📝 **Краткое резюме**: сразу после загрузки я автоматически составлю структурированное резюме документа.\n"
        "3. 🧠 **Поиск и ответы**: задавайте любые вопросы по тексту документов, я найду ответ с точным указанием страниц и источников.\n\n"
        "📌 **Доступные команды:**\n"
        "• `/help` — подробная инструкция\n"
        "• `/clear` — удалить ваши загруженные документы из векторной базы\n"
        "• `/status` — информация о системе\n\n"
        "Прикрепите и отправьте документ, чтобы начать!"
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "📖 **Инструкция по работе с ботом:**\n\n"
        "1️⃣ **Загрузка документа:**\n"
        "Отправьте документ как файл (поддерживаются PDF, DOCX, PPTX, TXT, MD).\n"
        "Docling разберет текст, заголовки и таблицы, разобьет документ на фрагменты и сохранит в векторную базу `pdf-bot`.\n\n"
        "2️⃣ **Автоматическое резюме:**\n"
        "Бот сразу пришлет главную тему, ключевые тезисы и подскажет вопросы по содержанию.\n\n"
        "3️⃣ **Вопросы по документу:**\n"
        "Просто напишите ваш вопрос обычным текстовым сообщением. Бот найдет нужный контекст и сформирует аргументированный ответ.\n\n"
        "4️⃣ **Очистка данных:**\n"
        "Команда `/clear` удалит ваши загруженные векторы из базы данных, если вы хотите начать с чистого листа."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    try:
        vector_store.delete_by_user(user_id)
        if user_id in user_chat_histories:
            user_chat_histories[user_id].clear()
        await update.message.reply_text(
            "🗑️ Ваши загруженные документы и история диалога успешно удалены из векторной базы.",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error in clear_handler for user {user_id}: {e}")
        await update.message.reply_text(f"⚠️ Ошибка при удалении данных: {e}")

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_text = (
        "📊 **Статус системы:**\n"
        f"• Индекс Pinecone: `{config.PINECONE_INDEX_NAME}` (1536 dim)\n"
        f"• Модель эмбеддингов: `{config.EMBEDDING_MODEL}`\n"
        f"• LLM Модель: `{config.LLM_MODEL}`\n"
        "• Парсер: `Docling DocumentConverter & HybridChunker`\n"
        "• Состояние: 🟢 Активен и готов к обработке документов."
    )
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    document = message.document
    if not document:
        return

    filename = document.file_name or "document.pdf"
    user_id = update.effective_user.id

    # Проверка формата
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        await message.reply_text(
            f"⚠️ Формат `{ext}` пока не поддерживается.\n"
            f"Пожалуйста, загрузите файл одного из форматов: {', '.join(SUPPORTED_EXTENSIONS)}",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    status_msg = await message.reply_text(
        f"📥 Получен файл **{filename}**.\nНачинаю загрузку...",
        parse_mode=ParseMode.MARKDOWN
    )

    temp_path = FileStorage.get_temp_filepath(filename, user_id)

    try:
        # Скачиваем файл из Telegram
        file_obj = await document.get_file()
        await file_obj.download_to_drive(str(temp_path))

        async def update_status(text: str):
            try:
                await status_msg.edit_text(text)
            except Exception:
                pass

        # Запускаем пайплайн обработки
        result = await ingestion_pipeline.run(
            file_path=temp_path,
            document_name=filename,
            user_id=user_id,
            status_callback=update_status
        )

        # Отправляем подтверждение и резюме
        success_header = (
            f"✅ **Файл успешно обработан и проиндексирован!**\n"
            f"📄 Документ: `{result['document_name']}`\n"
            f"📑 Страниц: {result['pages_count']} | Фрагментов (чанков): {result['chunks_count']}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        await status_msg.edit_text(success_header, parse_mode=ParseMode.MARKDOWN)

        # Отправляем структурированное резюме отдельным сообщением
        summary_text = result["summary"]
        await message.reply_text(summary_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error handling document {filename} for user {user_id}: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ Ошибка при обработке документа: {e}\n"
            "Пожалуйста, убедитесь, что файл не поврежден и повторите попытку."
        )
    finally:
        FileStorage.cleanup_file(temp_path)

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    user_query = message.text.strip()
    if user_query.startswith("/"):
        return  # Команды обрабатываются отдельными хэндлерами

    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)

    # Получаем историю сообщений для данного пользователя
    if user_id not in user_chat_histories:
        user_chat_histories[user_id] = []
    history = user_chat_histories[user_id]

    try:
        # Выполняем RAG поиск и генерацию ответа
        result = rag_pipeline.run(
            question=user_query,
            user_id=user_id,
            chat_history=history
        )

        answer_text = result["answer"]
        sources = result.get("sources", [])

        # Если найдены источники, добавим блок с ссылками на первоисточники
        if sources:
            sources_lines = []
            seen_sources = set()
            for src in sources:
                doc = src.get("document", "Документ")
                page = src.get("page")
                page_str = f", стр. {page}" if page else ""
                heading = src.get("headings")
                head_str = f" ({heading})" if heading else ""
                key = f"{doc}{page_str}{head_str}"
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources_lines.append(f"• `{doc}`{page_str}{head_str}")

            if sources_lines:
                answer_text += "\n\n📚 **Источники:**\n" + "\n".join(sources_lines[:4])

        # Сохраняем в историю диалога
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": result["answer"]})
        if len(history) > 6:
            user_chat_histories[user_id] = history[-6:]

        await message.reply_text(answer_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error handling message from user {user_id}: {e}", exc_info=True)
        await message.reply_text(
            f"⚠️ Произошла ошибка при поиске ответа: {e}",
            parse_mode=ParseMode.DEFAULT
        )
