import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from detect_function_dual import detect_image_dual
from detect_function import detect_image

# Переключатель для выбора функции детекции
use_dual_function = True  # Установите в False для использования стандартной функции детекции

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
def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=16)
    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        confidence = result["conf"]
        class_id = int(result["cls"])
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        color = default_colors[class_id % len(default_colors)]

        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Подготавливаем текст метки
        label = f"{class_name}: {confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_background = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]

        # Рисуем фон для текста
        draw.rectangle(text_background, fill=color)

        # Рисуем текст метки
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    return image

# Функция для обработки нескольких файлов
def process_files(files, gallery_state):
    if not files:
        return gallery_state, "Файлы не предоставлены."

    detection_function = detect_image_dual if use_dual_function else detect_image

    # Устанавливаем пороги внутри кода
    conf_thres = 0.4
    iou_thres = 0.4

    for file_path in tqdm(files, desc="Обработка изображений"):
        file_path = Path(file_path)  # Преобразуем каждый файл в объект Path
        print(f"Обрабатываемый файл: {file_path}")

        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file_path).convert("RGB")
            results = detection_function(
                weight_path,
                file_path,
                device="cpu",
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
            detected_image = draw_boxes(image, results)
            gallery_state.append(detected_image)

    return gallery_state, None

def reset_interface():
    return [], None

# Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Интерфейс обнаружения объектов</h1>")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Загрузите изображения",
                file_count="multiple",  # Разрешаем загружать несколько файлов
                type="filepath",  # Используем 'filepath', чтобы получить пути к файлам
            )
            submit_button = gr.Button("Обработать")

        with gr.Column():
            gallery_state = gr.State([])  # Для хранения изображений
            output_image = gr.Gallery(label="Галерея изображений")

    submit_button.click(
        process_files,
        inputs=[file_input, gallery_state],
        outputs=[output_image, file_input],
    )

    demo.load(fn=reset_interface, outputs=[gallery_state, file_input])

demo.launch(share=True)
