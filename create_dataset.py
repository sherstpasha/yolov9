# Установите библиотеку Pillow, если она еще не установлена
# pip install pillow

import os
import shutil
import glob

def process_images(input_dir, weights_path, labels_dir):
    # Создаем или очищаем директорию для изображений
    image_dir = 'images'
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)
    
    # Копируем файлы в директорию для изображений
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(image_dir, filename)
        shutil.copy(input_path, output_path)
    
    # Шаг 2: Запуск скрипта detect.py с указанием пути к весам и изображениям
    os.system(f"python detect.py --weights {weights_path} --conf 0.6 --line-thickness 2 --source {image_dir} --device 0 --save-txt --save-conf --imgsz 640")
    
    # Шаг 3: Перемещение результатов в соответствующие папки
    base_path = './runs/detect'  # путь может быть другим в зависимости от настройки
    exp_folders = glob.glob(os.path.join(base_path, 'exp*'))
    exp_numbers = [int(folder.split('exp')[-1]) for folder in exp_folders if folder.split('exp')[-1].isdigit()]
    max_exp = max(exp_numbers) if exp_numbers else None
    
    if max_exp is not None:
        latest_folder = os.path.join(base_path, f'exp{max_exp}')
    else:
        latest_folder = os.path.join(base_path, 'exp')
    
    # Создаем директорию для лейблов, если она не существует
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # Перемещаем файлы лейблов в указанную папку labels
    label_files = glob.glob(os.path.join(latest_folder, '*.txt'))
    for label_file in label_files:
        shutil.move(label_file, labels_dir)
    
    # Перемещаем файлы изображений в папку images
    image_files = glob.glob(os.path.join(latest_folder, '*.jpg'))
    for image_file in image_files:
        shutil.move(image_file, image_dir)
    
    print(f"Всего изображений: {len(image_files)}")
    print(f"Всего лейблов: {len(label_files)}")

# Пример вызова функции
input_dir = r'C:\Users\user\Desktop\reports_orig'  # укажите путь к вашей папке с изображениями
weights_path = r"C:\Users\user\Desktop\ddata\exp3\weights\best.pt"  # укажите путь к вашему файлу весов
labels_dir = r'C:\Users\user\Desktop\labels'  # укажите путь к папке, куда сохранять лейблы
process_images(input_dir, weights_path, labels_dir)
