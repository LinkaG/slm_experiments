# Фреймворк для экспериментов с малыми языковыми моделями

Этот фреймворк предоставляет гибкую систему для проведения экспериментов с малыми языковыми моделями (smolLM2, Qwen3) и различными ретриверами.

## Структура проекта

```
slm_experiments/
├── configs/               # Конфигурационные файлы Hydra
│   ├── model/            # Конфигурации моделей
│   ├── retriever/        # Конфигурации ретриверов
│   └── dataset/          # Конфигурации датасетов
├── src/
│   ├── data/             # Обработка датасетов и интеграция с S3
│   ├── models/           # Реализации моделей
│   ├── retrievers/       # Реализации ретриверов
│   ├── experiment/       # Запуск экспериментов и метрики
│   └── utils/            # Общие утилиты
├── tests/                # Тесты
└── scripts/              # Вспомогательные скрипты
```

## Установка

### Требования

- Python 3.10+ (рекомендуется 3.10-3.13)
- Poetry (менеджер зависимостей)
- Git

### Установка Poetry

Если Poetry не установлен:

```bash
# Установка через pip
pip3 install poetry

# Или через официальный установщик
curl -sSL https://install.python-poetry.org | python3 -
```

Если poetry установлен в пользовательскую дикерторию, то до вызова poetry использовать команду:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Настройка проекта

1. **Клонирование репозитория:**
```bash
git clone <repository-url>
cd slm_experiments
```

2. **Установка зависимостей:**
```bash
# Установка всех зависимостей
poetry install

# Или только основных (без PyTorch для экономии места)
poetry install --no-dev
```

3. **Активация виртуального окружения:**
```bash
# Активация shell
poetry shell

# Или запуск команд через poetry run
poetry run python script.py
```

4. **Настройка переменных окружения:**
```bash
# Создание файла .env (если нужен S3)
cp .env.example .env
# Отредактируйте .env, добавив учетные данные S3

# Настройка ClearML
cp clearml.conf ~/.clearml.conf
# Отредактируйте ~/.clearml.conf, добавив учетные данные ClearML
```

5. **Настройка сервера ClearML (при локальном запуске):**
```bash
poetry run clearml-server
```

### Альтернативная установка (без Poetry)

Если вы предпочитаете pip:

```bash
# Создание виртуального окружения
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Работа с данными

### Локальные данные

Проект поддерживает работу с локальными данными без необходимости настройки S3:

1. **Natural Questions (NQ)**:
   - Данные уже конвертированы и находятся в `data/nq/`
   - Train: `data/nq/nq_converted_train.json` (80 примеров)
   - Eval: `data/nq/nq_converted_eval.json` (20 примеров)

2. **Конвертация дополнительных данных**:
   ```bash
   # Конвертация с ограничением
   poetry run python convert_nq_data.py \
     --input data/nq/NQ-open.dev.merged.jsonl \
     --output data/nq/nq_converted.json \
     --max-items 1000 \
     --split
   ```

3. **Тестирование загрузки данных**:
   ```bash
   poetry run python test_local_data.py
   ```

### S3 данные (опционально)

Если у вас есть доступ к S3:

1. **Настройка AWS credentials**:
   ```bash
   cp .env.example .env
   # Отредактируйте .env, добавив:
   # AWS_ACCESS_KEY_ID=your_key
   # AWS_SECRET_ACCESS_KEY=your_secret
   # AWS_REGION=us-east-1
   ```

2. **Загрузка данных в S3**:
   ```bash
   aws s3 cp data/nq/train.json s3://your-bucket/datasets/nq/train.json
   aws s3 cp data/nq/eval.json s3://your-bucket/datasets/nq/eval.json
   ```

## Руководство по проведению экспериментов

### Подготовка к эксперименту

1. **Выбор компонентов**:
   - Определите модель (smolLM2 или Qwen3) и её размер
   - Выберите датасет (NQ или SimpleQA)
   - Выберите тип ретривера

2. **Проверка данных**:
   - Убедитесь, что данные доступны в S3 по указанным путям в конфигах
   - Проверьте наличие AWS credentials в `.env`
   - При необходимости очистите кэш данных (`.cache/datasets/`)

3. **Настройка окружения**:
   - Проверьте доступность GPU и объем памяти
   - Для больших моделей (>1.7B) убедитесь в наличии минимум 16GB GPU памяти
   - При необходимости настройте параметры оптимизации памяти в конфиге модели

### Режимы экспериментов

Фреймворк поддерживает три режима проведения экспериментов:

1. **Без контекста** (no_context):
   - Оценка способности модели отвечать на вопросы без дополнительного контекста
   - Проверка базовых знаний модели
   ```bash
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq experiment_mode=no_context
   ```

2. **С идеальным контекстом** (oracle_context):
   - Использование ground truth контекста из датасета
   - Оценка верхней границы производительности модели
   ```bash
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq experiment_mode=oracle_context
   ```

3. **С ретривером** (retriever_context):
   - Стандартный режим с использованием ретривера
   - Оценка производительности всего пайплайна
   ```bash
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq experiment_mode=retriever_context
   ```

### Запуск эксперимента

1. **Базовый запуск**:
   ```bash
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq
   ```

2. **Запуск с полной конфигурацией**:
   ```bash
   poetry run python -m src.cli run-experiment \
     model=smollm2_135m \
     dataset=local_nq \
     retriever=bm25 \
     experiment.name=my_custom_experiment
   ```

3. **Изменение параметров на лету**:
   ```bash
   poetry run python -m src.cli run-experiment \
     model=smollm2_135m \
     model.temperature=0.9 \
     model.batch_size=16
   ```

### Мониторинг и анализ

1. **Отслеживание прогресса**:
   - Откройте ClearML UI (http://localhost:8080 для локального сервера)
   - Найдите ваш эксперимент по имени
   - Следите за метриками в реальном времени

2. **Анализ результатов**:
   - Метрики сохраняются в `outputs/<experiment_name>/results.json`
   - Графики и сравнения доступны в ClearML UI
   - Token Recall показывает качество ответов модели
   - Детальная статистика использования памяти в `outputs/<experiment_name>/memory_usage.json`

3. **Мониторинг памяти**:
   - Отслеживание CPU и GPU памяти для модели и ретривера
   - Пиковое использование памяти
   - График потребления памяти в ClearML UI
   - Автоматическая очистка памяти каждые 100 примеров

3. **Отладка проблем**:
   - Проверьте логи в `outputs/<experiment_name>/`
   - При OOM ошибках уменьшите batch_size или включите оптимизации памяти
   - При проблемах с данными проверьте кэш и S3 доступ

### Сравнение экспериментов

1. **Запуск серии экспериментов**:
   ```bash
   # Сравнение разных размеров модели
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq
   poetry run python -m src.cli run-experiment model=smollm2_360m dataset=local_nq
   poetry run python -m src.cli run-experiment model=smollm2_1.7b dataset=local_nq
   ```

2. **Анализ результатов**:
   - Используйте ClearML для сравнения метрик между экспериментами
   - Сравнивайте Token Recall для разных конфигураций
   - Учитывайте размер индекса ретривера при анализе

### Добавление новых конфигураций

1. **Создание конфига модели**:
   - Скопируйте существующий конфиг из `configs/model/`
   - Измените параметры под ваши нужды
   - Сохраните с новым именем

2. **Настройка эксперимента**:
   - При необходимости добавьте новые параметры в `configs/config.yaml`
   - Создайте новые конфиги ретриверов в `configs/retriever/`
   - Документируйте изменения

### Советы и рекомендации

1. **Оптимизация памяти**:
   - Для моделей >1.7B используйте `load_in_8bit: true`
   - Настройте `batch_size` в зависимости от размера модели
   - Включите `gradient_checkpointing` для очень больших моделей

2. **Воспроизводимость**:
   - Всегда указывайте имя эксперимента
   - Фиксируйте все параметры в конфигах
   - Сохраняйте версии используемых моделей

3. **Производительность**:
   - Используйте кэширование данных
   - Правильно выбирайте batch_size
   - При возможности используйте flash attention

## Добавление новых компонентов

### Добавление новой модели
1. Создайте новый класс модели в `src/models/`
2. Реализуйте интерфейс `BaseModel`
3. Добавьте конфигурацию в `configs/model/`

### Добавление нового ретривера
1. Создайте новый класс ретривера в `src/retrievers/`
2. Реализуйте интерфейс `BaseRetriever`
3. Добавьте конфигурацию в `configs/retriever/`

### Добавление нового датасета
1. Создайте новый класс датасета в `src/data/`
2. Реализуйте интерфейс `BaseDataset`
3. Добавьте конфигурацию в `configs/dataset/`

## Полезные команды Poetry

```bash
# Просмотр установленных зависимостей
poetry show

# Обновление зависимостей
poetry update

# Добавление новой зависимости
poetry add package-name

# Добавление dev зависимости
poetry add --group dev package-name

# Запуск команды в виртуальном окружении
poetry run python script.py

# Активация shell
poetry shell

# Просмотр информации о проекте
poetry info

# Экспорт зависимостей в requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

## Структура проекта

```
slm_experiments/
├── data/                    # Данные
│   └── nq/                 # Natural Questions
│       ├── NQ-open.dev.merged.jsonl.zip  # Исходный архив
│       ├── NQ-open.dev.merged.jsonl      # Распакованный файл
│       ├── nq_converted_train.json       # Train данные
│       └── nq_converted_eval.json        # Eval данные
├── configs/                # Конфигурации Hydra
│   ├── model/             # Конфигурации моделей
│   ├── retriever/         # Конфигурации ретриверов
│   └── dataset/           # Конфигурации датасетов
├── src/                   # Исходный код
│   ├── data/              # Обработка датасетов
│   ├── models/            # Реализации моделей
│   ├── retrievers/        # Реализации ретриверов
│   ├── experiment/        # Запуск экспериментов и метрики
│   └── utils/             # Общие утилиты
├── tests/                 # Тесты
├── pyproject.toml         # Poetry конфигурация
├── convert_nq_data.py     # Скрипт конвертации данных
└── test_local_data.py     # Тест загрузки данных
```
