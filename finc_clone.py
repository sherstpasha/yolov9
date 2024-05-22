import os
import hashlib

def get_file_hash(file_path):
    """Возвращает хеш содержимого файла."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_duplicate_labels(labels_dir):
    # Проверяем, существует ли директория
    if not os.path.exists(labels_dir):
        print(f"Директория {labels_dir} не существует.")
        return
    
    # Получаем список всех файлов лейблов
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # Создаем словарь для хранения хешей файлов
    hash_dict = {}
    
    duplicates = []
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        file_hash = get_file_hash(label_path)
        
        if file_hash in hash_dict:
            duplicates.append((hash_dict[file_hash], label_file))
        else:
            hash_dict[file_hash] = label_file
    
    if duplicates:
        print("Найдены файлы с одинаковой разметкой:")
        for original, duplicate in duplicates:
            print(f"{duplicate} является дубликатом {original}")
    else:
        print("Дубликаты не найдены.")

# Пример вызова функции
labels_dir = r"C:\Users\user\Desktop\reports_detect_dataset — копия\labels"
  # укажите путь к вашей папке с лейблами
check_duplicate_labels(labels_dir)