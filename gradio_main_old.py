import gradio as gr
import zipfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yolov9
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from tqdm import tqdm

# Загрузка модели YOLOv9
model = yolov9.load(
    r"C:\Users\pasha\OneDrive\Рабочий стол\yolo_weights\yolo_word_detectino21.pt",
    device="cpu",
)
model.conf = 0.25  # Порог уверенности NMS
model.iou = 0.45  # Порог IoU NMS


def draw_boxes(image, boxes, confidences, class_ids, class_names):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{class_names[int(class_id)]} {confidence:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_location = [x1, y1 - text_size[3]]
        if text_location[1] < 0:
            text_location[1] = y1 + text_size[3]
        draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill="red")
        draw.text((x1, y1 - text_size[3]), text, fill="white", font=font)
    return image


def process_file(file):
    if file is None:
        return None

    file_path = Path(file.name)
    class_names = model.names  # Получаем названия классов
    if file_path.suffix in [".jpg", ".jpeg", ".png"]:
        # Если файл - изображение
        image = Image.open(file)
        results = model(np.array(image))
        boxes = results.pred[0][:, :4].cpu().numpy()  # Получаем координаты боксов
        confidences = results.pred[0][:, 4].cpu().numpy()  # Получаем уверенности
        class_ids = (
            results.pred[0][:, 5].cpu().numpy()
        )  # Получаем идентификаторы классов
        detected_image = draw_boxes(image, boxes, confidences, class_ids, class_names)
        return detected_image
    elif file_path.suffix in [".zip"]:
        try:
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall("temp_images")
            detected_results = []
            csv_data = []
            image_files = list(Path("temp_images").glob("*"))
            for img_path in tqdm(image_files, desc="Обработка изображений"):
                if img_path.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(img_path)
                    results = model(np.array(image))
                    boxes = (
                        results.pred[0][:, :4].cpu().numpy()
                    )  # Получаем координаты боксов
                    confidences = (
                        results.pred[0][:, 4].cpu().numpy()
                    )  # Получаем уверенности
                    class_ids = (
                        results.pred[0][:, 5].cpu().numpy()
                    )  # Получаем идентификаторы классов
                    detected_image = draw_boxes(
                        image, boxes, confidences, class_ids, class_names
                    )
                    detected_results.append(detected_image)
                    # Добавляем информацию о боксах в CSV данные
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        csv_data.append(
                            [
                                img_path.name,
                                class_names[int(class_id)],
                                confidence,
                                *box,
                            ]
                        )

            csv_df = pd.DataFrame(
                csv_data,
                columns=["Filename", "Class", "Confidence", "X1", "Y1", "X2", "Y2"],
            )

            # Открытие окна выбора пути для сохранения CSV файла
            root = tk.Tk()
            root.withdraw()
            root.lift()  # Поднятие окна поверх других
            root.attributes("-topmost", True)
            default_filename = f"{file_path.stem}_results.csv"
            save_path = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )

            if save_path:
                csv_df.to_csv(save_path, index=False)
                return None
            else:
                return None
        except Exception as e:
            print(f"Произошла ошибка при обработке архива: {str(e)}")
            return None
    else:
        print("Неподдерживаемый формат файла.")
        return None


# Gradio интерфейс
with gr.Blocks() as demo:
    with gr.Row():
        file_input = gr.File(
            label="Загрузите изображение или архив", file_count="single"
        )

    output_image = gr.Image(label="Детектированное изображение")

    file_input.change(process_file, inputs=file_input, outputs=[output_image])

# Запуск интерфейса с параметром share=True
demo.launch(share=False)
