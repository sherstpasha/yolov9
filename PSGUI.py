import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QWidget,
    QApplication,
)
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont
import zipfile
from pathlib import Path
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
    r"C:\Users\pasha\OneDrive\Рабочий стол\yolo_weights\yolo_word_detectino21.pt"
)


class MainApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("YOLO Detection")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        self.loadButton = QPushButton("Load Image or Zip", self)
        self.loadButton.clicked.connect(self.loadFile)
        layout.addWidget(self.loadButton)

        self.confSlider = QSlider(Qt.Horizontal)
        self.confSlider.setRange(0, 100)
        self.confSlider.setValue(25)
        self.confSlider.setTickInterval(5)
        self.confSlider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Confidence Threshold"))
        layout.addWidget(self.confSlider)

        self.iouSlider = QSlider(Qt.Horizontal)
        self.iouSlider.setRange(0, 100)
        self.iouSlider.setValue(45)
        self.iouSlider.setTickInterval(5)
        self.iouSlider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("IoU Threshold"))
        layout.addWidget(self.iouSlider)

        self.processButton = QPushButton("Process", self)
        self.processButton.clicked.connect(self.processFile)
        layout.addWidget(self.processButton)

        self.setLayout(layout)

    def draw_boxes(self, image, results):
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

    def loadFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or Zip",
            "",
            "Images (*.png *.xpm *.jpg);;Zip Files (*.zip)",
            options=options,
        )
        if fileName:
            self.filePath = fileName
            self.imageLabel.setPixmap(QtGui.QPixmap(fileName))

    def processFile(self):
        if not hasattr(self, "filePath"):
            QtWidgets.QMessageBox.warning(self, "Error", "No file loaded")
            return

        conf_thres = self.confSlider.value() / 100
        iou_thres = self.iouSlider.value() / 100
        file_path = Path(self.filePath)
        detection_function = detect_image_dual if use_dual_function else detect_image

        if file_path.suffix in [".jpg", ".jpeg", ".png"]:
            try:
                results = detection_function(
                    weight_path,
                    file_path,
                    device="cpu",
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                )
                image = Image.open(file_path)
                detected_image = self.draw_boxes(image, results)
                detected_image.save("output.jpg")
                self.imageLabel.setPixmap(QtGui.QPixmap("output.jpg"))
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Error processing image: {str(e)}"
                )
        elif file_path.suffix in [".zip"]:
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall("temp_images")
                detected_results = []
                csv_data = []
                image_files = list(Path("temp_images").glob("*"))
                for img_path in tqdm(image_files, desc="Processing Images"):
                    if img_path.suffix in [".jpg", ".jpeg", ".png"]:
                        try:
                            results = detection_function(
                                weight_path,
                                img_path,
                                device="cpu",
                                conf_thres=conf_thres,
                                iou_thres=iou_thres,
                            )
                            image = Image.open(img_path)
                            detected_image = self.draw_boxes(image, results)
                            detected_results.append(detected_image)
                            for result in results:
                                box = result["bbox"]
                                confidence = result["conf"]
                                class_id = result["cls"]
                                csv_data.append(
                                    [img_path.name, int(class_id), confidence, *box]
                                )
                        except Exception as e:
                            print(f"Error processing image {img_path}: {str(e)}")

                csv_df = pd.DataFrame(
                    csv_data,
                    columns=["Filename", "Class", "Confidence", "X1", "Y1", "X2", "Y2"],
                )
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save CSV", "", "CSV Files (*.csv)"
                )
                if save_path:
                    csv_df.to_csv(save_path, index=False)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Cancelled", "CSV save cancelled"
                    )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Error processing archive: {str(e)}"
                )
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Unsupported file format")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainApp = MainApp()
    mainApp.show()
    sys.exit(app.exec_())
