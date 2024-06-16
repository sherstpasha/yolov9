## Описание решения

В данном репозитории описано решение команды Центр ИИ для задачи "Определение и классификация дефектов сварных швов с помощью ИИ" 

Модель Yolov9 https://drive.google.com/file/d/18VjW3AztQILnj_6YFHMPRuawGR-48f9A/view?usp=drive_link
Модель ViT https://drive.google.com/drive/folders/1ajqhnrkoffJSgxMBJ7z88ge6tCXyHt5V?usp=sharing
Ноутбук для запуска бота и получения предсказания https://colab.research.google.com/drive/1GGM9bq4PJP4FiPiIhU5tpzh-AO71mipf?usp=sharing


Воспроизвести решение можно двумя способами:
1) Собрать докер контейнер и запустить телеграм бота, передав ему токен;
2) Запустить телеграм бота с помощью сервиса google colab, используя этот ноутбук.


## Запуск Телеграм бота через контейнер

### 1. Склонируйте репозиторий:

```bash
git clone https://github.com/sherstpasha/yolov9
```

### 2. Соберите Docker контейнер:

```bash
docker build -t container_name .
```

Например:
 ```bash
docker build -t yolo9 .
```

### 3. Запустите контейнер с графическим процессором (монтируйте папку с данными):

```bash
docker run --gpus all -it -v /путь/к/вашей/папке:/workspace/mounted_folder yolo9
```

Например:
 ```bash
docker run --gpus all -it -v C:/Users/user/Desktop/data_and_weight:/workspace/mounted_folder yolo9
```

### 4. Запустите телеграмм бот, выполнив команду:
```bash
python main.py --weights path/to/weights.pt --token your_telegram_api_token
```

Например:
```bash
python main.py --weights mounted_folder/best_weights.pt --token 1234567890:BBIJ_YDFW_QGhhfrLMRXYOHqqrZyUMvwXUy
```

## Запуск телеграмм бота через ноутбук

