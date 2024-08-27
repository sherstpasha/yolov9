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

# Функция для рисования боксов на изображении
def draw_boxes_on_image(image, true_boxes, pred_boxes, true_color="#00FF00", pred_color="#FF0000"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=16)
    width, height = image.size

    # Рисуем истинные боксы (предполагается, что координаты нормализованы)
    for box in true_boxes:
        class_id = int(box["cls"])

        # Преобразуем относительные координаты в абсолютные
        x_center, y_center, bbox_width, bbox_height = box["bbox"]
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)
        
        # Проверка, что боксы попадают в видимую область изображения
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"Warning: True box for class {class_names[class_id]} is out of image bounds: ({x1}, {y1}), ({x2}, {y2})")
            continue

        # Вывод координат бокса в консоль для проверки
        print(f"True box for class {class_names[class_id]}: ({x1}, {y1}), ({x2}, {y2})")

        # Рисуем прямоугольник и текст для истинного бокса
        draw.rectangle([x1, y1, x2, y2], outline=true_color, width=3)
        label = f"True: {class_names[class_id]}"

        # Добавление фона под текст для лучшей видимости
        text_bbox = draw.textbbox((x1, y1 - 10), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=true_color)
        draw.text((x1, y1 - 10), label, fill="white", font=font)

    # Рисуем предсказанные боксы (предполагается, что координаты уже абсолютные)
    for box in pred_boxes:
        class_id = int(box["cls"])

        # Используем абсолютные координаты напрямую
        x1, y1, x2, y2 = map(int, box["bbox"])

        # Проверка, что боксы попадают в видимую область изображения
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"Warning: Predicted box for class {class_names[class_id]} is out of image bounds: ({x1}, {y1}), ({x2}, {y2})")
            continue

        # Вывод координат бокса в консоль для проверки
        print(f"Predicted box for class {class_names[class_id]}: ({x1}, {y1}), ({x2}, {y2})")

        # Рисуем прямоугольник и текст для предсказанного бокса
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=3)
        label = f"Pred: {class_names[class_id]}"

        # Добавление фона под текст для лучшей видимости
        text_bbox = draw.textbbox((x1, y1 - 10), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=pred_color)
        draw.text((x1, y1 - 10), label, fill="white", font=font)

    return image


# Функция обработки изображений из указанной папки
def process_images_in_folder(input_folder, output_folder, conf_thres=0.4, iou_thres=0.45, use_dual_function=True):
    image_folder = Path(input_folder) / "images"
    label_folder = Path(input_folder) / "labels"
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
        label_path = label_folder / (image_path.stem + ".txt")
        
        # Чтение истинных боксов (если необходимо)
        true_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    true_boxes.append({"cls": class_id, "bbox": bbox})

        # Получение предсказанных боксов
        pred_boxes = detection_function(
            weight_path,
            image_path,
            device="cpu",
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        print(pred_boxes)
        
        # Рисуем истинные и предсказанные боксы на одном изображении
        combined_image = image.copy()
        combined_image = draw_boxes_on_image(combined_image, true_boxes, pred_boxes)
        combined_image.save(output_combined_folder / image_path.name)
        
    print("Обработка завершена.")

# Использование функции
input_folder = r"C:\Users\user\Desktop\Проволока\снимки для КРИТБИ\yoloimages\all_data"  # Путь до папки с папками images и labels
output_folder = r"C:\Users\user\Desktop\Проволока\снимки для КРИТБИ\yoloimages\out"  # Путь для сохранения изображений с боксами

process_images_in_folder(input_folder, output_folder)