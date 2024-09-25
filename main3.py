# -*- coding: utf-8 -*-
import os
import shutil
import io
import zipfile
from pathlib import Path
import torch
import nest_asyncio
from telegram import Update, InputFile, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image, ImageDraw, ImageFont
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.dataloaders import LoadImages, LoadCV2Image
from utils.torch_utils import select_device, smart_inference_mode
from detect_function import detect_image
from detect_function_dual import detect_image_dual


# Пути к файлам весов для каждой модели
WEIGHTS = {
    "welding": "mounted_folder/weld_model.pt",  # заменить на фактический путь к модели
    "aluminum": "mounted_folder/bars_model.pt",  # заменить на фактический путь к модели
    "tvel": "mounted_folder/vtel_model.pt"  # заменить на фактический путь к модели
}

# Названия моделей для вывода пользователю и выбор функции детекции
MODEL_NAMES_AND_FUNCTIONS = {
    "welding": ("Дефекты сварки", detect_image),  # используем detect_image
    "aluminum": ("Трещины на алюминиевой чушке", detect_image_dual),  # используем detect_image_dual
    "tvel": ("Дефект навивки ТВЭЛ", detect_image_dual)  # используем detect_image_dual
}

# Переменные для отслеживания режима пользователя и confidence
USER_MODES = {}
USER_CONFIDENCE = {}

# Функция для создания клавиатуры с выбором модели
def get_model_selection_keyboard():
    keyboard = [
        [KeyboardButton("Дефекты сварки"), KeyboardButton("Трещины на алюминии"), KeyboardButton("Дефекты на ТВЭЛ элементе")]
    ]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    USER_MODES[update.effective_user.id] = None
    USER_CONFIDENCE[update.effective_user.id] = 0.1  # Значение по умолчанию
    await update.message.reply_text('Привет! Выберите модель для детекции дефектов:', reply_markup=get_model_selection_keyboard())

# Обработчик выбора модели
async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    model_choice = update.message.text.lower()

    if model_choice == "дефекты сварки":
        USER_MODES[user_id] = "welding"
    elif model_choice == "трещины на алюминии":
        USER_MODES[user_id] = "aluminum"
    elif model_choice == "дефекты на твэл элементе":
        USER_MODES[user_id] = "tvel"
    else:
        await update.message.reply_text('Пожалуйста, выберите корректную модель из предложенных вариантов.')
        return

    model_name = MODEL_NAMES_AND_FUNCTIONS[USER_MODES[user_id]][0]
    await update.message.reply_text(f"Вы выбрали: {model_name}. Теперь отправьте фото для анализа.")

# Функция для обработки изображений с выбранной моделью
def process_images_with_model_selection(user_id, image_dir, conf_thres):
    model_choice = USER_MODES.get(user_id, "welding")  # По умолчанию модель сварки
    model_path = WEIGHTS.get(model_choice, WEIGHTS["welding"])
    
    # Выбираем функцию детекции в зависимости от выбранной модели
    detect_function = MODEL_NAMES_AND_FUNCTIONS[model_choice][1]
    
    # Запускаем детекцию с использованием выбранной функции
    results = detect_function(weights=model_path, source=image_dir, conf_thres=conf_thres, device="cpu")
    return results

# Функция для рисования боксов на изображении
def draw_boxes(image_bytes, bboxes):
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for bbox in bboxes:
        box = bbox['bbox']
        conf = bbox['conf']
        cls = bbox['cls']
        label = f'Class {cls} {conf:.2f}'
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), label, fill="red", font=font)

    output = io.BytesIO()
    image.save(output, format='JPEG')
    output.seek(0)
    return output

# Обработчик отправленных изображений
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    conf_thres = USER_CONFIDENCE.get(user_id, 0.1)

    try:
        if update.message.photo:
            await update.message.reply_text('Изображение получено, подождите немного.')
            file = await context.bot.get_file(update.message.photo[-1].file_id)
            photo_bytes = await file.download_as_bytearray()

            # Обрабатываем фото с выбранной моделью
            image_dir = 'temp_image'
            os.makedirs(image_dir, exist_ok=True)
            with open(os.path.join(image_dir, 'photo.jpg'), 'wb') as img_file:
                img_file.write(photo_bytes)

            detection_results = process_images_with_model_selection(user_id, image_dir, conf_thres)
            results = {'photo.jpg': detection_results}

            # Отправляем фото с нарисованными боксами
            img_bytes = draw_boxes(photo_bytes, detection_results)
            await context.bot.send_photo(chat_id=update.message.chat_id, photo=img_bytes, caption="Результаты детекции")

            # Удаляем временные файлы
            shutil.rmtree(image_dir)

        # Обработка других типов файлов (архивы, документы) по аналогии...
    
    except Exception as e:
        await update.message.reply_text(f'Произошла ошибка: {e}')

# Запуск бота
def main() -> None:
    nest_asyncio.apply()

    # Вставьте ваш токен от BotFather
    token = '7287622548:AAGBEwjd5nhQS-XhGv4sa6Ihc06LOfZlHM4'

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model_selection))
    application.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO, handle_file))

    application.run_polling()
