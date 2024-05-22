import os

def remove_images_without_labels(images_dir, labels_dir):
    # Проверяем, существуют ли директории
    if not os.path.exists(images_dir):
        print(f"Директория {images_dir} не существует.")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Директория {labels_dir} не существует.")
        return
    
    # Получаем список всех файлов изображений и лейблов
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # Создаем множество базовых имен файлов лейблов (без расширения)
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    
    # Удаляем изображения, для которых нет соответствующего файла лейбла
    removed_count = 0
    for image_file in image_files:
        image_basename = os.path.splitext(image_file)[0]
        if image_basename not in label_basenames:
            os.remove(os.path.join(images_dir, image_file))
            removed_count += 1
    
    print(f"Удалено изображений: {removed_count}")

# Пример вызова функции
images_dir = r"C:\Users\user\Desktop\reports_detect_dataset\images"  # укажите путь к вашей папке с изображениями
labels_dir = r"C:\Users\user\Desktop\reports_detect_dataset\labels"  # укажите путь к вашей папке с лейблами
remove_images_without_labels(images_dir, labels_dir)
