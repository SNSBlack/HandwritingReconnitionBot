import os
import sys
import io
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from inference import OCREngine

_bot_dir     = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_bot_dir)

_env_loaded = False
for _folder in [_bot_dir, _project_dir]:
    for _name in [".env", ".env.example"]:
        _env_path = os.path.join(_folder, _name)
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            print(f"✅ Загружен файл токена: {_env_path}")
            _env_loaded = True
            break
    if _env_loaded:
        break

if not _env_loaded:
    print("⚠️  Файл .env не найден. Убедитесь что файл .env существует в папке bot/ или в корне проекта.")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ocr_engine: OCREngine | None = None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветственное сообщение."""
    user = update.effective_user
    text = (
        f"Привет, {user.first_name}! 👋\n\n"
        "Я умею распознавать **рукописный русский текст** с фотографий.\n\n"
        "📸 Просто отправь мне фото с рукописным текстом, "
        "и я пришлю тебе его печатную версию.\n\n"
        "Команды:\n"
        "/start — это сообщение\n"
        "/help  — помощь\n"
        "/info  — информация о модели"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Помощь."""
    text = (
        "📖 *Как пользоваться ботом:*\n\n"
        "1. Сфотографируй рукописный текст\n"
        "2. Отправь фото мне в чат\n"
        "3. Получи распознанный текст\n\n"
        "💡 *Советы для лучшего распознавания:*\n"
        "• Хорошее освещение\n"
        "• Текст ровно, без сильного наклона\n"
        "• Чёткий фокус на тексте\n"
        "• Тёмные чернила на белой бумаге\n\n"
        "⚡ Бот работает на GPU для быстрого распознавания"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Информация о модели."""
    import torch
    mode = ocr_engine.mode if ocr_engine else "не инициализирован"
    device = str(ocr_engine.device) if ocr_engine else "—"

    gpu_info = "—"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"{gpu_name} ({vram:.1f} GB)"

    text = (
        f"🤖 *Информация о системе*\n\n"
        f"Режим OCR: `{mode}`\n"
        f"Устройство: `{device}`\n"
        f"GPU: `{gpu_info}`\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #функция обрабатывания входящего фото: скачивает → распознаёт → отвечает.
    message: Message = update.message
    user = update.effective_user

    logger.info(f"Фото от @{user.username} (id={user.id})")

    status_msg = await message.reply_text("🔍 Анализирую рукопись...")

    try:
        photo = message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        img_bytes = await file.download_as_bytearray()
        img_data = bytes(img_bytes)

        logger.info(f"Размер файла: {len(img_data) / 1024:.1f} KB")
        recognized_text = ocr_engine.recognize_from_bytes(img_data)

        logger.info(f"Результат: {repr(recognized_text)}")
        if recognized_text and recognized_text != "(текст не распознан)":
            

            reply = (
                f"📝 *Распознанный текст:*\n\n"
                f"`{recognized_text}`"
            )
        else:
            reply = (
                "😔 Не удалось распознать текст.\n\n"
                "Попробуйте:\n"
                "• Улучшить освещение\n"
                "• Сфотографировать ближе\n"
                "• Убедиться что текст читаем"
            )

        await status_msg.delete()
        await message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}", exc_info=True)
        await status_msg.edit_text(
            "❌ Произошла ошибка при обработке фото.\n"
            "Попробуйте ещё раз или отправьте другое изображение."
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #функция обработки документа
    doc = update.message.document

    #проверка изображения
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text(
            "📎 Пожалуйста, отправьте изображение (JPG, PNG, WEBP)."
        )
        return

    status_msg = await update.message.reply_text("🔍 Анализирую рукопись...")

    try:
        file = await context.bot.get_file(doc.file_id)
        img_bytes = await file.download_as_bytearray()
        img_data = bytes(img_bytes)

        recognized_text = ocr_engine.recognize_from_bytes(img_data)

        if recognized_text and recognized_text != "(текст не распознан)":
            reply = f"📝 *Распознанный текст:*\n\n`{recognized_text}`"
        else:
            reply = "😔 Не удалось распознать текст."

        await status_msg.delete()
        await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Ошибка обработки документа: {e}", exc_info=True)
        await status_msg.edit_text("❌ Ошибка при обработке файла.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Реакция на текстовые сообщения."""
    await update.message.reply_text(
        "📸 Отправьте мне фото с рукописным текстом, и я его распознаю!\n"
        "Используйте /help для подробной инструкции."
    )

def main():
    global ocr_engine

    parser = argparse.ArgumentParser(description="Telegram OCR бот")
    parser.add_argument(
        "--mode", choices=["trocr", "easyocr", "crnn"], default="trocr",
        help="Режим OCR: trocr (по умолчанию), easyocr, crnn"
    )
    parser.add_argument(
        "--checkpoint", default=r"D:\data sets\cheks_trocr\best_hf",
        help="Путь к модели (для trocr — папка HuggingFace, для crnn — .pth файл)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Устройство: cuda, cpu, cuda:0, ..."
    )
    parser.add_argument(
        "--token", default=None,
        help="Telegram Bot Token (или задайте TELEGRAM_BOT_TOKEN в .env)"
    )
    args = parser.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Токен бота не найден!")
        print("   Укажите --token <TOKEN> или задайте TELEGRAM_BOT_TOKEN в .env файле")
        print("   Получите токен у @BotFather в Telegram")
        sys.exit(1)

    print(f"🚀 Запускаю OCR бот в режиме: {args.mode}")
    ocr_engine = OCREngine(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
    )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("info", cmd_info))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("✅ Бот запущен. Нажмите Ctrl+C для остановки.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()