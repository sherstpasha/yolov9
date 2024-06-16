# @title Запуск бота
import os
import zipfile
import nest_asyncio
import io
import shutil
from telegram import (
    Update,
    InputFile,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from PIL import Image, ImageDraw, ImageFont
from detect_function import detect_image
from telegram.error import Forbidden
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTImageProcessor
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Telegram bot for image processing.")
    parser.add_argument(
        "--token", type=str, required=True, help="Telegram bot token from BotFather."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to the weights file for object detection.",
    )

    return parser.parse_args()


# Словарь классов и цветов
CLASS_NAMES = {0: "adj", 1: "int", 2: "geo", 3: "pro", 4: "non"}
CLASS_COLORS = {0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple"}

# Переменные для отслеживания режима и порога confidence
USER_MODES = {}
USER_CONFIDENCE = {}

# Максимальный размер файла в байтах (1 ГБ)
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024

# Максимальные размеры изображения
MAX_IMAGE_SIZE = 1000, 1000

import os
from PIL import Image


def preprocess_image(image, image_processor):
    transforms = Compose(
        [
            Resize((image_processor.size["height"], image_processor.size["width"])),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    return transforms(image)


def process_images(image_dir, conf_thres, weights):
    device_yolo = 0 if torch.cuda.is_available() else "cpu"
    device_vit = "cuda" if torch.cuda.is_available() else "cpu"

    # Результаты детекции с использованием YOLO
    results = detect_image(
        weights=weights,
        source=image_dir,
        conf_thres=conf_thres,
        device=device_yolo,
    )

    # Инициализация модели классификации
    output_dir = "sherstpasha/ViT_welding_defects"
    model = ViTForImageClassification.from_pretrained(output_dir)
    model.to(device_vit)
    image_processor = ViTImageProcessor.from_pretrained(output_dir)
    print(results)
    # Обработка результатов детекции
    for result in results:
        image_path = result["path"]
        bbox = result["bbox"]
        with Image.open(image_path) as img:
            # Вырезаем область по боксу
            cropped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            # Препроцессинг изображения для классификации
            processed_img = preprocess_image(cropped_img, image_processor)
            processed_img = processed_img.unsqueeze(0)  # Добавляем batch dimension

            # Классификация изображения
            model.eval()
            with torch.no_grad():
                outputs = model(processed_img.to(device_vit))
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            # Обновление класса и конфиденциальности
            result["cls"] = predicted_class_idx
    print(results)
    return results


# Вспомогательные функции
def load_image(image_path):
    return Image.open(image_path)


# Функция для распаковки архива
def unzip_file(file_data):
    image_files = {}
    with zipfile.ZipFile(io.BytesIO(file_data), "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.lower().endswith(
                (
                    ".bmp",
                    ".dng",
                    ".jpeg",
                    ".jpg",
                    ".mpo",
                    ".png",
                    ".tif",
                    ".tiff",
                    ".webp",
                    ".pfm",
                )
            ):
                image_files[file] = zip_ref.read(file)
    return image_files


# Функция для изменения размера изображений
def resize_image(image):
    original_size = image.size
    image.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return image, original_size


# Функция для рисования боксов на изображении
def draw_boxes(image_bytes, bboxes):
    image = Image.open(io.BytesIO(image_bytes))
    image, original_size = resize_image(image)
    draw = ImageDraw.Draw(image)
    try:
        # Увеличение размера шрифта и установка кириллического шрифта
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    scale_x = image.size[0] / original_size[0]
    scale_y = image.size[1] / original_size[1]

    if not bboxes:
        draw.text((10, 10), "", fill="red", font=font)
    else:
        for bbox in bboxes:
            box = [
                int(coord * scale_x) if i % 2 == 0 else int(coord * scale_y)
                for i, coord in enumerate(bbox["bbox"])
            ]
            conf = bbox["conf"]
            cls = bbox["cls"]
            color = CLASS_COLORS.get(cls, "red")
            label = CLASS_NAMES.get(cls, "Unknown")
            draw.rectangle(box, outline=color, width=3)
            # Добавляем контур к тексту
            text = f"{label} {conf:.2f}"
            text_size = draw.textbbox((0, 0), text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            x, y = box[0], box[1] - text_height
            draw.rectangle([x, y, x + text_width, y + text_height], fill=color)
            text_color = "black" if color == "yellow" else "white"
            draw.text((x, y), text, fill=text_color, font=font)

    output = io.BytesIO()
    image.save(output, format="JPEG")
    output.seek(0)
    return output


# Функция для нормализации координат
def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return x_center, y_center, box_width, box_height


# Функция для создания архива с результатами
def create_results_archive(results, image_files, original_filename):
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_path, bboxes in results.items():
            img_bytes = image_files[image_path]
            with Image.open(io.BytesIO(img_bytes)) as img:
                img = resize_image(img)[0]
                width, height = img.size
            txt_content = ""
            for bbox in bboxes:
                x_center, y_center, box_width, box_height = normalize_bbox(
                    bbox["bbox"], width, height
                )
                conf = bbox["conf"]
                cls = bbox["cls"]
                txt_content += f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {conf:.6f}\n"
            txt_filename = os.path.splitext(image_path)[0] + ".txt"
            archive.writestr(txt_filename, txt_content)
    archive_buffer.seek(0)
    archive_buffer.name = f"{os.path.splitext(original_filename)[0]}_labels.zip"
    return archive_buffer


# Функция для создания CSV с результатами
def create_results_csv(results, image_files):
    csv_buffer = io.StringIO()
    csv_buffer.write("filename;class_id;rel_x;rel_y;width;height\n")
    for image_path, bboxes in results.items():
        img_bytes = image_files[image_path]
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = resize_image(img)[0]
            width, height = img.size
        for bbox in bboxes:
            x_center, y_center, box_width, box_height = normalize_bbox(
                bbox["bbox"], width, height
            )
            cls = bbox["cls"]
            csv_buffer.write(
                f"{os.path.basename(image_path)};{cls};{x_center:.6f};{y_center:.6f};{box_width:.6f};{box_height:.6f}\n"
            )
    csv_buffer.seek(0)
    return io.BytesIO(csv_buffer.getvalue().encode("utf-8"))


# Функция для создания клавиатуры
def get_keyboard():
    keyboard = [[KeyboardButton("Выбрать пороговое значение")]]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    USER_MODES[update.effective_user.id] = None
    USER_CONFIDENCE[update.effective_user.id] = 0.1  # Значение по умолчанию
    await update.message.reply_text(
        "Привет! Отправьте мне фото, изображение или архив с изображениями для обработки.",
        reply_markup=get_keyboard(),
    )


# Функция для подсчета дефектов по классам
def count_defects(bboxes):
    counts = {CLASS_NAMES[cls]: 0 for cls in CLASS_NAMES}
    for bbox in bboxes:
        cls = bbox["cls"]
        counts[CLASS_NAMES[cls]] += 1
    total = sum(counts.values())
    return counts, total


# Обработчик полученных файлов и сообщений
async def handle_file(
    update: Update, context: ContextTypes.DEFAULT_TYPE, weights: str
) -> None:
    user_id = update.message.from_user.id
    conf_thres = USER_CONFIDENCE.get(user_id, 0.1)

    try:
        if update.message.photo:
            await update.message.reply_text("Изображение получено, подождите немного.")
            file = await context.bot.get_file(update.message.photo[-1].file_id)
            photo_bytes = await file.download_as_bytearray()

            # Обрабатываем фото
            image_dir = "temp_image"
            os.makedirs(image_dir, exist_ok=True)
            with open(os.path.join(image_dir, "photo.jpg"), "wb") as img_file:
                img_file.write(photo_bytes)

            detection_results = process_images(image_dir, conf_thres, weights)
            results = {"photo.jpg": detection_results}
            counts, total = count_defects(detection_results)

            # Отправляем фото с нарисованными боксами и статистику
            img_bytes = draw_boxes(photo_bytes, detection_results)
            if total == 0:
                await context.bot.send_photo(
                    chat_id=update.message.chat_id,
                    photo=img_bytes,
                    caption="Дефектов не найдено.",
                )
            else:
                caption = (
                    "\n".join([f"{cls}: {count}" for cls, count in counts.items()])
                    + f"\nВсего: {total}"
                )
                await context.bot.send_photo(
                    chat_id=update.message.chat_id, photo=img_bytes, caption=caption
                )

            # Удаляем временные файлы
            shutil.rmtree(image_dir)

        elif update.message.document:
            if update.message.document.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    "Файл слишком большой. Максимальный размер файла - 1 ГБ."
                )
                return

            if update.message.document.mime_type.startswith("image/"):
                await update.message.reply_text(
                    "Изображение получено, подождите немного."
                )
                file = await context.bot.get_file(update.message.document.file_id)
                photo_bytes = await file.download_as_bytearray()

                # Обрабатываем фото
                image_dir = "temp_image"
                os.makedirs(image_dir, exist_ok=True)
                with open(os.path.join(image_dir, "photo.jpg"), "wb") as img_file:
                    img_file.write(photo_bytes)

                detection_results = process_images(image_dir, conf_thres, weights)
                results = {"photo.jpg": detection_results}
                counts, total = count_defects(detection_results)

                # Отправляем фото с нарисованными боксами и статистику
                img_bytes = draw_boxes(photo_bytes, detection_results)
                if total == 0:
                    await context.bot.send_photo(
                        chat_id=update.message.chat_id,
                        photo=img_bytes,
                        caption="Дефектов не найдено.",
                    )
                else:
                    caption = (
                        "\n".join([f"{cls}: {count}" for cls, count in counts.items()])
                        + f"\nВсего: {total}"
                    )
                    await context.bot.send_photo(
                        chat_id=update.message.chat_id, photo=img_bytes, caption=caption
                    )

                # Удаляем временные файлы
                shutil.rmtree(image_dir)

            elif update.message.document.mime_type == "application/zip":
                await update.message.reply_text("Архив получен, подождите немного.")
                file = await context.bot.get_file(update.message.document.file_id)
                file_data = await file.download_as_bytearray()
                original_filename = update.message.document.file_name

                # Распаковываем архив
                image_files = unzip_file(file_data)

                if not image_files:
                    await update.message.reply_text(
                        "Не найдено изображений поддерживаемых форматов в архиве."
                    )
                    return

                # Сохраняем изображения в временную директорию для обработки
                image_dir = "images"
                if os.path.exists(image_dir):
                    shutil.rmtree(image_dir)
                os.makedirs(image_dir)
                for img_name, img_data in image_files.items():
                    with open(
                        os.path.join(image_dir, os.path.basename(img_name)), "wb"
                    ) as img_file:
                        img_file.write(img_data)

                # Обрабатываем изображения
                detection_results = process_images(image_dir, conf_thres, weights)

                # Преобразуем результаты в нужный формат
                results = {}
                for result in detection_results:
                    image_path = os.path.basename(result["path"])
                    if image_path not in results:
                        results[image_path] = []
                    results[image_path].append(
                        {
                            "bbox": result["bbox"],
                            "conf": result["conf"],
                            "cls": result["cls"],
                        }
                    )

                # Формируем и отправляем CSV с результатами
                results_csv = create_results_csv(results, image_files)
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=InputFile(results_csv, filename="submission.csv"),
                )

                # Удаляем временные файлы
                shutil.rmtree(image_dir)

            else:
                await update.message.reply_text(
                    "Пожалуйста, отправьте фото, изображение или архив с изображениями."
                )

        else:
            await update.message.reply_text(
                "Пожалуйста, отправьте фото, изображение или архив с изображениями."
            )

    except Forbidden:
        print(f"Bot was blocked by the user: {update.message.chat_id}")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")


# Обработчик текстовых сообщений
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text
    if text == "Выбрать пороговое значение":
        await update.message.reply_text(
            "Введите значение порога от 1 до 99:", reply_markup=ReplyKeyboardRemove()
        )
        USER_MODES[user_id] = "set_threshold"
    elif USER_MODES.get(user_id) == "set_threshold":
        try:
            value = int(text)
            if 1 <= value <= 99:
                USER_CONFIDENCE[user_id] = value / 100.0
                await update.message.reply_text(
                    f"Установлен порог confidence: {USER_CONFIDENCE[user_id]}",
                    reply_markup=get_keyboard(),
                )
                USER_MODES[user_id] = None
            else:
                await update.message.reply_text(
                    "Пожалуйста, введите значение от 1 до 99."
                )
        except ValueError:
            await update.message.reply_text(
                "Пожалуйста, введите корректное числовое значение."
            )
    else:
        await update.message.reply_text(
            "Пожалуйста, отправьте фото, изображение или архив с изображениями."
        )


def main() -> None:
    args = parse_args()  # Parse command line arguments
    nest_asyncio.apply()  # Обходим проблему с уже запущенным event loop

    # Используем полученные аргументы
    application = ApplicationBuilder().token(args.token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    application.add_handler(
        MessageHandler(
            filters.Document.ALL | filters.PHOTO,
            lambda update, context: handle_file(update, context, args.weights),
        )
    )

    application.run_polling()


if __name__ == "__main__":
    main()
