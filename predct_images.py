import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil
from detect_function_dual import detect_image_dual
from detect_function import detect_image

# Путь к весам модели
weight_path = r"C:\Users\user\Desktop\crispi_defects\data\exp3\weights\best.pt"

# Заданные в коде названия классов
class_names = ["Вмятина", "На проволоке", "Накол"]

# Расширенный список цветов для каждого класса
default_colors = [
    "#000000",  # Чёрный
    "#00008B",  # Тёмно-синий
    "#8B0000",  # Тёмно-красный
    "#006400",  # Тёмно-зелёный
    "#4B0082",  # Индиго
    "#2F4F4F",  # Тёмный серо-зелёный
    "#8B008B",  # Тёмная магента
    "#8B4513",  # Седло-коричневый
    "#800000",  # Бордовый
    "#483D8B",  # Тёмный серо-синий
    "#4682B4",  # Стальной синий
    "#556B2F",  # Тёмный оливковый
    "#6A5ACD",  # Сланцевый синий
    "#2E8B57",  # Морская волна
    "#A0522D",  # Сиенна
]

# Функция для рисования легенды на изображении
def draw_legend(image, class_names, colors, font):
    draw = ImageDraw.Draw(image, "RGBA")
    padding = 10
    
    # Вычисляем высоту легенды и ширину на основе размеров текста
    legend_height = len(class_names) * (font.getbbox(class_names[0])[3] + padding) + padding
    legend_width = max(font.getbbox(name)[2] - font.getbbox(name)[0] for name in class_names) + padding * 4 + 40
    
    # Координаты верхнего правого угла
    x1 = image.width - legend_width - padding
    y1 = padding
    x2 = image.width - padding
    y2 = y1 + legend_height
    
    # Рисуем фон для легенды
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 200))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        text_position = (x1 + padding * 2 + 30, y1 + padding + i * (font.getbbox(name)[3] + padding))
        color_position = (x1 + padding, text_position[1] + (font.getbbox(name)[3] - font.getbbox(name)[1] - 20) // 2)

        # Рисуем цветной квадрат
        draw.rectangle([color_position, (color_position[0] + 20, color_position[1] + 20)], fill=color)

        # Рисуем текст
        draw.text(text_position, name, fill=(0, 0, 0, 255), font=font)
    
    return image

# Функция для рисования предсказанных боксов на изображении
def draw_boxes_on_image(image, pred_boxes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=16)
    width, height = image.size

    # Рисуем предсказанные боксы (предполагается, что координаты уже абсолютные)
    for box in pred_boxes:
        class_id = int(box["cls"])
        confidence = box["conf"]

        # Используем абсолютные координаты напрямую
        x1, y1, x2, y2 = map(int, box["bbox"])

        # Проверка, что боксы попадают в видимую область изображения
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"Warning: Predicted box for class {class_names[class_id]} is out of image bounds: ({x1}, {y1}), ({x2}, {y2})")
            continue

        # Рисуем прямоугольник для предсказанного бокса
        color = default_colors[class_id % len(default_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Подготавливаем текст с именем класса и вероятностью
        label = f"{class_names[class_id]}: {confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 10), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 10), label, fill="white", font=font)

    # Добавляем легенду на изображение
    image = draw_legend(image, class_names, default_colors, font)

    return image


# Функция обработки изображений из указанной папки
def process_images_in_folder(input_folder, output_folder, conf_thres=0.4, iou_thres=0.45, use_dual_function=True):
    image_folder = Path(input_folder) / "images"
    output_combined_folder = Path(output_folder) / "combined_boxes"
    
    # Создаем выходные папки, если их нет
    output_combined_folder.mkdir(parents=True, exist_ok=True)
    
    # Выбор функции детекции
    detection_function = detect_image_dual if use_dual_function else detect_image
    
    # Обрабатываем все изображения в папке
    for image_path in image_folder.glob("*.*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = Image.open(image_path).convert("RGB")
        
        # Получение предсказанных боксов
        pred_boxes = detection_function(
            weight_path,
            image_path,
            device="cpu",
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        # Рисуем предсказанные боксы на изображении
        combined_image = image.copy()
        combined_image = draw_boxes_on_image(combined_image, pred_boxes)
        combined_image.save(output_combined_folder / image_path.name)
        
    print("Обработка завершена.")

# Использование функции
input_folder = r"C:\Users\user\Desktop\crispi_defects\data\test"  # Путь до папки с папками images и labels
output_folder = r"C:\Users\user\Desktop\Проволока\снимки для КРИТБИ\yoloimages\out_test"  # Путь для сохранения изображений с боксами

process_images_in_folder(input_folder, output_folder)
