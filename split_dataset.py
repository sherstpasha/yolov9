import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7901, val_ratio=0.19, test_ratio=0.0201):
    # Убедитесь, что пропорции суммируются до 1.0
    #assert train_ratio + val_ratio + test_ratio == 1.0, "Train, val, and test ratios must sum to 1.0"

    # Создание директорий для train, val и test
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)

    # Список всех файлов изображений
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    # Перемешивание списка изображений
    random.shuffle(images)
    
    # Определение количества файлов для каждой части
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    # Разделение файлов
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Функция для копирования файлов изображений и аннотаций
    def copy_files(file_list, subset):
        for image_file in file_list:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            try:
                shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, subset, 'images', image_file))
                shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, subset, 'labels', label_file))
            except Exception as e:
                print(f"Could not copy {image_file} or {label_file}: {e}")

    # Копирование файлов
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')


# Пример использования скрипта
images_dir = r"C:\Users\user\Desktop\ddatas\data\images"
labels_dir = r"C:\Users\user\Desktop\ddatas\data\labels"
output_dir = r"C:\Users\user\Desktop\ddatas\datadatadata"
split_dataset(images_dir, labels_dir, output_dir)