## Использование

1. Сначала установите Docker и драйверы NVIDIA GPU на вашем компьютере. 
Для этого проекта используется контейнер с версией CUDA 12.4.1.

2. Склонируйте репозиторий:

```bash
git clone https://github.com/sherstpasha/yolov9
```

3. Соберите Docker контейнер:

```bash
docker build -t yolo9 .
```

4. Запустите контейнер с графическим процессором (при необходимости монтируйте папку с данными):

```bash
docker run --gpus all -it -v /путь/к/вашей/папке:/workspace/mounted_folder yolo9
```

Например:
 ```bash
docker run --gpus all -it -v C:/Users/user/Desktop/data_and_weight:/workspace/mounted_folder yolo9
```


5. Теперь, когда вы находитесь в контейнере, вы можете запустить `train.py` с помощью следующей команды, чтобы начать обучение модели:

```bash
python train.py --batch 16 --epochs 300 --img 640 --device 0 --min-items 0 --data path/to/data.yaml --weights path/to/weights.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml
```

Например:
```bash
python train.py --batch 32 --epochs 5 --img 640 --device 0 --min-items 0 --data mounted_folder/data/data.yaml  --weights mounted_folder/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml --workers 0
```

6. После окончания обучения в докере результат обучения сохраняется в папке "runs/train/exp". Чтобы перенести ее на вашу систему, выполните следующую команду (не в докере, а в терминале вашей системы) `docker cp`, указав путь к папке в контейнере и путь на вашей системе, куда вы хотите скопировать файл.

```bash
docker cp <container_id>:runs/train/exp .
```

Например:

```bash
docker cp <container_id>:runs/train/exp .
```