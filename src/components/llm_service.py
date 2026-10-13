from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.config import config
from src.utils.logger import logger

class LLMService:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.model = model or config.LLM_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_summary(self, document_name: str, document_text: str) -> str:
        """
        Генерирует структурированное резюме загруженного документа.
        """
        # Ограничим текст первыми ~15000 символов, если документ огромный, для качественного summary
        sample_text = document_text[:16000]

        prompt = f"""Ты — интеллектуальный ассистент по анализу документов.
Пользователь загрузил документ: "{document_name}".

Ниже приведен текст документа:
---
{sample_text}
---

Сформируй понятное, структурированное резюме на русском языке в следующем формате (используй Markdown):

📄 **Документ:** {document_name}

🎯 **Основная тема и цель документа:**
(1-2 предложения, о чем этот документ)

📌 **Ключевые разделы и положения:**
- Пункт 1
- Пункт 2
- Пункт 3

💡 **Главные выводы/факты:**
(Краткий список важных фактов, цифр или выводов)

❓ **Что вы можете спросить у меня по этому документу:**
- [Пример вопроса 1]
- [Пример вопроса 2]
- [Пример вопроса 3]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты профессиональный аналитик документов. Отвечай структурированно и наглядно."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content or "Не удалось сформировать резюме."
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"⚠️ Документ успешно проиндексирован, но возникла ошибка при создании краткого резюме: {e}"

    def answer_question(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Отвечает на вопрос пользователя, опираясь на контекст найденных чанков.
        """
        if not context_chunks:
            return "К сожалению, в загруженных документах не найдено информации, относящейся к вашему вопросу. Попробуйте переформулировать вопрос или загрузить соответствующий документ."

        # Формируем блок контекста с указанием источников и метаданных
        context_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Документ")
            page = meta.get("page_number")
            page_info = f", стр. {page}" if page else ""
            heading = meta.get("headings")
            heading_info = f" (Раздел: {heading})" if heading else ""
            
            chunk_text = chunk.get("text", "").strip()
            context_blocks.append(f"[Источник {i}: {doc_name}{page_info}{heading_info}]\n{chunk_text}")

        context_str = "\n\n".join(context_blocks)

        system_prompt = """Ты — интеллектуальный помощник, отвечающий на вопросы исключительно на основе предоставленного контекста документов.
Правила:
1. Отвечай точно, полно и опирайся ТОЛЬКО на предоставленный контекст.
2. Если в контексте нет ответа, честно скажи: "В предоставленных документах нет информации по данному вопросу." Не придумывай факты.
3. В конце ответа или по тексту указывай ссылки на источники (например: [Документ.pdf, стр. 2] или [Источник 1]).
4. Форматируй ответ красиво с использованием Markdown."""

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            # Ограничиваем последние 4 сообщения истории для поддержания диалога
            for msg in chat_history[-4:]:
                messages.append(msg)

        user_content = f"""Контекст из документов:
{context_str}

Вопрос пользователя: {question}"""

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content or "Не удалось сгенерировать ответ."
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"⚠️ Произошла ошибка при обращении к языковой модели: {e}"
