## Обучение

Ниже описан процесс обучения модели yolov9 в контейнере. Для запуска необходимо установить Docker.

1. Склонируйте репозиторий:

```bash
git clone https://github.com/sherstpasha/yolov9
```

2. Соберите Docker контейнер:

```bash
docker build -t yolo9 .
```

3. Запустите контейнер с графическим процессором (монтируйте папку с данными):

```bash
docker run --gpus all -it -v /путь/к/вашей/папке:/workspace/mounted_folder yolo9
```

Например:
 ```bash
docker run --gpus all -it -v C:/Users/user/Desktop/data_and_weight:/workspace/mounted_folder yolo9
```


4. Теперь, когда вы находитесь в контейнере, вы можете запустить `train.py` с помощью следующей команды, чтобы начать обучение модели:

```bash
python train.py --batch 16 --epochs 300 --img 640 --device 0 --min-items 0 --data path/to/data.yaml --weights path/to/weights.pt --cfg models/detect/config_file.yaml --hyp path/to/hyp_file.yaml --project path/to/project_folder
```

Например:
```bash
python train.py --batch 32 --epochs 5 --img 640 --device 0 --min-items 0 --data mounted_folder/data/data.yaml --project mounted_folder/ --weights mounted_folder/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml --project mounted_folder/
```

5. После окончания обучения в докере результат обучения сохраняется в папке, которая была указана как "--project path/to/project_folder"

Запуск 
