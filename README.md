## Описание решения

В данном репозитории описано решение команды Центр ИИ для задачи "Определение и классификация дефектов сварных швов с помощью ИИ" 

В нашем решении мы используем комбинированный подход из двух эффективных моделей машинного обучения: YOLOv9 и ViT. YOLOv9 определяет дефекты, а ViT занимается их классификацией. Эта система работает через удобного Telegram-бота, предоставляя простой и эффективный способ загрузки фотографий и получения обратной связи.

Обученные модели можно получить по следующим ссылкам:
1. [Модель Yolov9](https://drive.google.com/file/d/18VjW3AztQILnj_6YFHMPRuawGR-48f9A/view?usp=drive_link);
2. [Модель ViT](https://drive.google.com/drive/folders/1ajqhnrkoffJSgxMBJ7z88ge6tCXyHt5V?usp=sharing).


Воспроизвести решение можно двумя способами:
1) Собрать докер контейнер и запустить телеграм бота, передав ему токен;
2) Запустить телеграм бота с помощью сервиса google colab, используя этот [ноутбук](https://colab.research.google.com/drive/1-SIQQNbmShljF5Zg5OS8bDlMHvthDNJW?usp=sharing);
3) Перейти на уже запущенного [телеграм бота](https://t.me/DefectDetectBot).

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

## Запуск телеграм бота через ноутбук

Для этого достаточно запустить [ноутбук](https://colab.research.google.com/drive/1-SIQQNbmShljF5Zg5OS8bDlMHvthDNJW?usp=sharing) и выполнить ячейку "ЗАПУСК БОТА", предварительно передав токен.


## Дополнительные материалы
Ноутбук для запуска телеграм бота и получения предсказания доступен по [этой ссылке](notebooks/WeldingDefectDetectYolov9_ViT.ipynb).

Ноутбук для обучения ViT доступен по [этой ссылке](notebooks/vit.ipynb).

Описание запуска обучения YOLOv9 доступно в [файле](YOLOV9_TRAIN.md).
