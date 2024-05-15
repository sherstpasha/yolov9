# Выбор базового образа с предустановленным Python
FROM python:3.9-slim

# Установка неинтерактивного режима для APT (автоматический выбор ответов по умолчанию)
ENV DEBIAN_FRONTEND=noninteractive

# Настройка временной зоны (пример для часовой зоны Москва)
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Обновление списка пакетов и установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    python3-opencv \
    # Удаление списков пакетов после установки для уменьшения размера образа
    && rm -rf /var/lib/apt/lists/*

# Копирование файла требований зависимостей Python в контейнер
COPY requirements.txt /workspace/requirements.txt

# Установка зависимостей Python из файла требований
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Копирование оставшегося исходного кода проекта в контейнер
COPY . /workspace

# Установка рабочей директории
WORKDIR /workspace

# Команда по умолчанию, запускающая оболочку bash
CMD ["bash"]
