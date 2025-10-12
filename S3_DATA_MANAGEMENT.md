# Управление данными в S3

Этот документ описывает работу с данными в S3 хранилище для проекта slm_experiments.

## 🔧 Настройка

### 1. Конфигурация .env

Убедитесь, что файл `.env` содержит корректные креденшиалы S3:

```bash
# S3 Storage Configuration
CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
CLEARML_S3_BUCKET=clearml-artifacts
CLEARML_S3_ACCESS_KEY=minio_admin_2024
CLEARML_S3_SECRET_KEY=Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0
CLEARML_S3_REGION=us-east-1
CLEARML_S3_PATH_STYLE=true
CLEARML_S3_VERIFY_SSL=false
```

### 2. Установка зависимостей

```bash
# Установка boto3 и python-dotenv
poetry add boto3 python-dotenv
```

## 📤 Загрузка данных в S3

### Скрипт: `upload_to_s3.py`

#### Базовое использование:

```bash
# Тест подключения к S3
poetry run python upload_to_s3.py --mode test

# Загрузка NQ данных
poetry run python upload_to_s3.py --mode nq

# Загрузка SimpleQA данных
poetry run python upload_to_s3.py --mode simple_qa

# Загрузка произвольного файла
poetry run python upload_to_s3.py --mode custom \
  --local-file data/my_data.json \
  --s3-key datasets/custom/my_data.json
```

#### Опции:

- `--mode`: Режим загрузки (test, nq, simple_qa, custom)
- `--data-dir`: Директория с данными (по умолчанию: data)
- `--local-file`: Локальный файл (для режима custom)
- `--s3-key`: S3 ключ (для режима custom)
- `--overwrite`: Перезаписывать существующие файлы
- `--bucket`: S3 bucket (переопределяет .env)

#### Структура загружаемых данных:

**NQ данные:**
- `data/nq/nq_full_dataset.json` → `datasets/nq/nq_full_dataset.json`
- `data/nq/nq_converted_eval.json` → `datasets/nq/eval.json`
- `data/nq/nq_converted_train.json` → `datasets/nq/train.json`

**SimpleQA данные:**
- `data/simple_qa/simple_qa_converted.json` → `datasets/simple_qa/simple_qa_converted.json`
- `data/simple_qa/simple_qa_converted.json` → `datasets/simple_qa/train.json`
- `data/simple_qa/simple_qa_converted.json` → `datasets/simple_qa/eval.json`

## 📥 Скачивание данных из S3

### Скрипт: `download_from_s3.py`

#### Базовое использование:

```bash
# Тест подключения к S3
poetry run python download_from_s3.py --mode test

# Просмотр содержимого bucket
poetry run python download_from_s3.py --mode browse

# Просмотр файлов с префиксом
poetry run python download_from_s3.py --mode browse --prefix datasets/

# Скачивание NQ данных
poetry run python download_from_s3.py --mode nq

# Скачивание SimpleQA данных
poetry run python download_from_s3.py --mode simple_qa

# Скачивание произвольного файла
poetry run python download_from_s3.py --mode custom \
  --s3-key datasets/nq/nq_full_dataset.json \
  --local-file data/downloaded_nq.json
```

#### Опции:

- `--mode`: Режим скачивания (test, browse, nq, simple_qa, custom)
- `--data-dir`: Директория для сохранения (по умолчанию: data)
- `--s3-key`: S3 ключ (для режима custom)
- `--local-file`: Локальный путь (для режима custom)
- `--prefix`: Префикс для поиска (для режима browse)
- `--overwrite`: Перезаписывать существующие файлы
- `--bucket`: S3 bucket (переопределяет .env)

## 🔍 Тестирование S3 подключения

### Скрипт: `test_s3_connection.py`

#### Базовое использование:

```bash
# Базовый тест подключения
poetry run python test_s3_connection.py

# Тест с проверкой операций с файлами
poetry run python test_s3_connection.py --test-operations

# Тест с проверкой файлов датасетов
poetry run python test_s3_connection.py --check-datasets

# Полный тест
poetry run python test_s3_connection.py --test-operations --check-datasets
```

#### Что проверяет:

1. **Креденшиалы** - корректность AWS ключей
2. **Доступ к bucket** - возможность работы с указанным bucket
3. **Список buckets** - доступные buckets
4. **Содержимое bucket** - файлы в bucket
5. **Операции с файлами** - создание/чтение/удаление (опционально)
6. **Файлы датасетов** - наличие файлов NQ и SimpleQA (опционально)

## 📊 Рабочий процесс

### 1. Обработка данных из S3 (рекомендуемый способ)

```bash
# Тест подключения к S3
poetry run python test_s3_connection.py

# Обработка NQ данных (скачать исходный файл, конвертировать, загрузить результат)
poetry run python process_s3_data.py --mode nq

# Обработка SimpleQA данных
poetry run python process_s3_data.py --mode simple_qa

# Обработка всех данных сразу
poetry run python process_s3_data.py --mode both

# Обработка без загрузки результатов в S3 (только локально)
poetry run python process_s3_data.py --mode both --no-upload
```

### 2. Ручная подготовка данных (альтернативный способ)

```bash
# Скачивание исходных файлов из S3
poetry run python download_from_s3.py --mode custom \
  --s3-key NQ-open.dev.merged.jsonl \
  --local-file data/nq/NQ-open.dev.merged.jsonl

poetry run python download_from_s3.py --mode custom \
  --s3-key simple_qa_test_set_with_documents.csv \
  --local-file data/simple_qa/simple_qa_test_set_with_documents.csv

# Конвертация NQ данных
poetry run python convert_nq_data.py \
  --input data/nq/NQ-open.dev.merged.jsonl \
  --output data/nq/nq_full_dataset.json

# Конвертация SimpleQA данных
poetry run python convert_simple_qa_data.py \
  --input data/simple_qa/simple_qa_test_set_with_documents.csv \
  --output data/simple_qa/simple_qa_converted.json

# Загрузка результатов в S3
poetry run python upload_to_s3.py --mode nq
poetry run python upload_to_s3.py --mode simple_qa
```

### 3. Использование в экспериментах

```bash
# Эксперимент с S3 данными
poetry run python run_experiment_simple.py dataset=rag_nq

# Эксперимент с локальными данными
poetry run python run_experiment_simple.py dataset=local_nq
```

### 4. Скачивание для локального использования

```bash
# Скачивание всех данных
poetry run python download_from_s3.py --mode nq
poetry run python download_from_s3.py --mode simple_qa

# Тест локальных данных
poetry run python test_local_data.py
```

## 🛠 Устранение проблем

### Ошибка подключения к S3

```bash
# Проверьте конфигурацию
poetry run python test_s3_connection.py

# Проверьте .env файл
cat .env | grep CLEARML_S3
```

### Ошибка доступа к bucket

```bash
# Проверьте права доступа
poetry run python test_s3_connection.py --test-operations
```

### Файлы не найдены

```bash
# Просмотрите содержимое bucket
poetry run python download_from_s3.py --mode browse

# Проверьте наличие файлов датасетов
poetry run python test_s3_connection.py --check-datasets
```

### Ошибка загрузки данных

```bash
# Проверьте существование локальных файлов
ls -la data/nq/
ls -la data/simple_qa/

# Попробуйте с --overwrite
poetry run python upload_to_s3.py --mode nq --overwrite
```

## 📁 Структура данных

### Локальная структура:

```
data/
├── nq/                           # Natural Questions
│   ├── NQ-open.dev.merged.jsonl  # Исходный файл
│   ├── nq_full_dataset.json      # Полный датасет
│   ├── nq_converted_eval.json    # Eval данные
│   └── nq_converted_train.json   # Train данные
├── simple_qa/                    # SimpleQA
│   ├── simple_qa_test_set_with_documents.csv  # Исходный файл
│   └── simple_qa_converted.json  # Конвертированный датасет
└── .cache/                       # Кэш данных
    └── datasets/
```

### S3 структура (реальная):

```
s3://clearml-artifacts/
├── NQ-open.dev.merged.jsonl                    # Исходный файл NQ
├── simple_qa_test_set_with_documents.csv        # Исходный файл SimpleQA
├── nq_full_dataset.json                         # Конвертированный NQ
├── nq_converted_eval.json                       # NQ eval данные
├── nq_converted_train.json                      # NQ train данные
├── simple_qa_converted.json                     # Конвертированный SimpleQA
├── simple_qa_train.json                         # SimpleQA train данные
└── simple_qa_eval.json                          # SimpleQA eval данные
```

## 💡 Советы

1. **Кэширование**: Данные автоматически кэшируются в `.cache/datasets/`
2. **Перезапись**: Используйте `--overwrite` для обновления существующих файлов
3. **Тестирование**: Всегда тестируйте подключение перед загрузкой данных
4. **Мониторинг**: Следите за логами для диагностики проблем
5. **Резервное копирование**: Регулярно создавайте резервные копии важных данных
