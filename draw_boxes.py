import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor

# Путь к весам модели
weight_path = r"C:\Users\user\Desktop\crispi_defects\data\exp3\weights\best.pt"

# Заданные в коде названия классов
class_names = ["Пузырь", "Трещина"]

# Расширенный список цветов для каждого класса с альфа-каналом (прозрачность)
default_colors = [
    (0, 0, 0, 128),        # Чёрный с прозрачностью
    (0, 0, 139, 128),      # Тёмно-синий с прозрачностью
    (139, 0, 0, 128),      # Тёмно-красный с прозрачностью
    (0, 100, 0, 128),      # Тёмно-зелёный с прозрачностью
    (75, 0, 130, 128),     # Индиго с прозрачностью
    (47, 79, 79, 128),     # Тёмный серо-зелёный с прозрачностью
    (139, 0, 139, 128),    # Тёмная магента с прозрачностью
    (139, 69, 19, 128),    # Седло-коричневый с прозрачностью
    (128, 0, 0, 128),      # Бордовый с прозрачностью
    (72, 61, 139, 128),    # Тёмный серо-синий с прозрачностью
    (70, 130, 180, 128),   # Стальной синий с прозрачностью
    (85, 107, 47, 128),    # Тёмный оливковый с прозрачностью
    (106, 90, 205, 128),   # Сланцевый синий с прозрачностью
    (46, 139, 87, 128),    # Морская волна с прозрачностью
    (160, 82, 45, 128),    # Сиенна с прозрачностью
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

# Функция для рисования истинных боксов на изображении
def draw_true_boxes_on_image(image, true_boxes, start_index=1):
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("arial.ttf", size=16)
    width, height = image.size

    # Рисуем истинные боксы (предполагается, что координаты нормализованы)
    for i, box in enumerate(true_boxes, start=start_index):
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

        # Используем цвет, соответствующий классу, с прозрачностью
        color = default_colors[class_id % len(default_colors)]
        
        # Рисуем более тонкий полупрозрачный прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{i}"  # Только номер

        # Центрирование текста по высоте бокса и размещение его слева
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1 - text_width - 5  # Размещаем текст левее бокса с отступом
        text_y = y1 + (bbox_height * height - text_height) / 2  # Центрируем текст по высоте бокса
        
        # Добавление прозрачного фона под текст для лучшей видимости
        text_bg_color = (color[0], color[1], color[2], 128)  # Прозрачный фон под текст
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], fill=text_bg_color)
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 128), font=font)  # Прозрачный белый текст

    # Добавляем легенду на изображение
    image = draw_legend(image, class_names, default_colors, font)

    return image, len(true_boxes)

# Функция обработки изображений из указанной папки
def process_images_in_folder(input_folder, output_folder, start_index=1):
    image_folder = Path(input_folder) / "images"
    label_folder = Path(input_folder) / "labels"
    output_combined_folder = Path(output_folder) / "combined_boxes"
    
    # Создаем выходные папки, если их нет
    output_combined_folder.mkdir(parents=True, exist_ok=True)
    
    current_index = start_index
    
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

        if true_boxes:
            # Рисуем истинные боксы на изображении и обновляем текущий индекс
            combined_image, num_boxes = draw_true_boxes_on_image(image, true_boxes, start_index=current_index)
            combined_image.save(output_combined_folder / image_path.name)
            current_index += num_boxes
        
    print("Обработка завершена. Последний индекс бокса:", current_index - 1)

# Использование функции
input_folder = r"C:\Users\user\Desktop\rusal_data\data\test"  # Путь до папки с папками images и labels
output_folder = r"C:\Users\user\Desktop\rusal_data\out"  # Путь для сохранения изображений с боксами

process_images_in_folder(input_folder, output_folder)
