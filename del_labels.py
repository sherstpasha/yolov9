import os

def remove_confidence_from_labels(labels_dir):
    # Проверяем, существует ли директория с лейблами
    if not os.path.exists(labels_dir):
        print(f"Директория {labels_dir} не существует.")
        return
    
    # Получаем все файлы лейблов в директории
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        
        # Читаем содержимое файла
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Убираем степень уверенности из каждого файла
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:
                parts.pop(-1)  # Удаляем последний элемент, который является степенью уверенности
            new_lines.append(" ".join(parts) + "\n")
        
        # Сохраняем отредактированный файл
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
    
    print(f"Обработано файлов лейблов: {len(label_files)}")

# Пример вызова функции
labels_dir = r"C:\Users\user\Desktop\reports_detect_dataset\labels"  # укажите путь к вашей папке с лейблами
remove_confidence_from_labels(labels_dir)
