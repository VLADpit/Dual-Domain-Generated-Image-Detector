#Dual-Domain AI-Generated Image Detector
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Проект для классификации изображений на **реальные** и **сгенерированные** (StyleGAN), созданный как итоговый проект яндекс лицея 
В основе лежит **двухдоменная нейронная сеть (Dual-Domain CNN)**, которая анализирует как пространственные признаки, так и частотные артефакты, характерные для генеративных моделей. Добавлена api обёртка через Fastapi и Docker 

## Идея и Архитектура
Генеративные модели часто оставляют специфические следы в высокочастотном спектре изображений, которые неочевидны для человеческого глаза, но легко детектируются сетью.

**Архитектура `DualDomainCNN`:**
1. **Пространственная часть:** Классическая двойная свёртка (VGG) для извлечения семантических и текстурных признаков из изображения
2. **Частотная часть:** Аналогичная свёрточная сеть, но принимающая на вход **логарифмический спектр мощности** (после преобразования Фурье и применения двустороннего высокочастотного фильтра BiHPF)
3. **Слияние:** Конкатенация признаков из обеих ветвей и финальный классификатор

```text
[ Input Image (RGB) ] 
       │
       ├─► [ Spatial Branch ] ─► Conv2D -> Pool -> Conv2D -> Pool -> AdaptiveAvgPool ─┐
       │    (Извлекает текстуры, семантику, артефакты наложения)                      │
       │                                                                              ├─► [ Fusion ] ─► [ Classifier (Real/Fake) ]
       └─► [ Frequency Branch ] ─► FFT -> LogMag -> BiHPF ─► Conv2D -> Pool ...  ─────┘
            (Ищет сеточные артефакты, аномалии в спектре )

## Установка
1. Клонируйте репозиторий и перейдите в него:
   ```bash
   git clone https://github.com/VLADpit/Dual-Domain-Generated-Image-Detector.git
   cd dual-domain-detector

## Запуск

### Вариант 1:Python
1. **Установка зависимостей:**
   ```bash
   pip install -r requirements.txt
   # Если нужна поддержка GPU, установите PyTorch отдельно:
   # pip install torch torchvision
   ```
2. **Подготовка весов**
   Поместите файл обученной модели в корень проекта:
   ```text
   best_model_dual_domain.pth
   ```

3. **Запуск сервера:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Вариант 2:Docker
1. **Подготовка весов:**
   Создайте папку `weights` и положите туда файл модели:
   ```text
   ./weights/best_model_dual_domain.pth
   ```

2. **Запуск через Docker Compose:**
   ```bash
   docker compose up --build -d
   ```


## Управление контейнером
| Задача | Команда |
|---|---|
| Посмотреть логи| `docker compose logs -f` |
| Перезапустить контейнер | `docker compose restart` |
| Перезапустить и пересобрать образ | `docker compose up --build -d` |
| Остановить и удалить контейнер | `docker compose down` |
