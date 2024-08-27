import gradio as gr
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tqdm import tqdm
from detect_function_dual import detect_image_dual
from detect_function import detect_image

# Переключатель для выбора функции детекции
use_dual_function = (
    True  # Установите в False для использования стандартной функции детекции
)

# Путь к весам модели
weight_path = (
    r"C:\Users\user\Desktop\crispi_defects\data\exp3\weights\best.pt"
)


# Функция для рисования боксов на изображении
def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        confidence = result["conf"]
        class_id = result["cls"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"Class {int(class_id)} {confidence:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_location = [x1, y1 - text_size[3]]
        if text_location[1] < 0:
            text_location[1] = y1 + text_size[3]
        draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill="red")
        draw.text((x1, y1 - text_size[3]), text, fill="white", font=font)
    return image


# Функция для обработки одного файла
def process_file(file, conf_thres, iou_thres):
    if file is None or not file.name:
        print("Файл не предоставлен")
        return None

    try:
        file_path = Path(file.name)
        detection_function = detect_image_dual if use_dual_function else detect_image

        print(
            f"Используется функция: {'detect_image_dual' if use_dual_function else 'detect_image'}"
        )
        print(f"Параметры: conf_thres={conf_thres}, iou_thres={iou_thres}")
        print(f"Обрабатываемый файл: {file_path}")

        if file_path.suffix in [".jpg", ".jpeg", ".png"]:
            try:
                print(f"Обработка изображения: {file_path}")
                results = detection_function(
                    weight_path,
                    file_path,
                    device="cpu",
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                )
                print(f"Результаты детекции: {results}")
                image = Image.open(file)
                detected_image = draw_boxes(image, results)
                return detected_image
            except Exception as e:
                print(f"Произошла ошибка при обработке изображения: {str(e)}")
                return None
        elif file_path.suffix in [".zip"]:
            try:
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall("temp_images")
                detected_results = []
                csv_data = []
                image_files = list(Path("temp_images").glob("*"))
                for img_path in tqdm(image_files, desc="Обработка изображений"):
                    if img_path.suffix in [".jpg", ".jpeg", ".png"]:
                        try:
                            print(f"Обработка изображения из архива: {img_path}")
                            results = detection_function(
                                weight_path,
                                img_path,
                                device="cpu",
                                conf_thres=conf_thres,
                                iou_thres=iou_thres,
                            )
                            print(
                                f"Результаты детекции для {img_path}: {results.keys()}"
                            )
                            image = Image.open(img_path)
                            detected_image = draw_boxes(image, results)
                            detected_results.append(detected_image)
                            # Добавляем информацию о боксах в CSV данные
                            for result in results:
                                box = result["bbox"]
                                confidence = result["conf"]
                                class_id = result["cls"]
                                csv_data.append(
                                    [
                                        img_path.name,
                                        int(class_id),
                                        confidence,
                                        *box,
                                    ]
                                )
                        except Exception as e:
                            print(
                                f"Произошла ошибка при обработке изображения {img_path}: {str(e)}"
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
                    print(f"Сохранение CSV файла в: {save_path}")
                    csv_df.to_csv(save_path, index=False)
                    return None
                else:
                    print("Сохранение CSV файла отменено")
                    return None
            except Exception as e:
                print(f"Произошла ошибка при обработке архива: {str(e)}")
                return None
        else:
            print("Неподдерживаемый формат файла.")
            return None
    except Exception as e:
        print(f"Произошла общая ошибка: {str(e)}")
        return None


def reset_interface():
    return None


# Gradio интерфейс
with gr.Blocks() as demo:
    with gr.Row():
        file_input = gr.File(
            label="Загрузите изображение или архив", file_count="single"
        )
        with gr.Column():
            conf_thres_input = gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
            iou_thres_input = gr.Slider(0, 1, value=0.45, label="IoU Threshold")

    output_image = gr.Image(label="Детектированное изображение")

    def process_file_wrapper(file, conf_thres, iou_thres):
        return process_file(file, conf_thres, iou_thres)

    file_input.change(
        process_file_wrapper,
        inputs=[file_input, conf_thres_input, iou_thres_input],
        outputs=[output_image],
    )

    file_input.clear(fn=reset_interface, inputs=None, outputs=[output_image])

# Запуск интерфейса с параметром share=False
demo.launch(share=True)
