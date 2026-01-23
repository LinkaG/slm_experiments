# Режим работы без ClearML

Фреймворк поддерживает работу без ClearML и MinIO. В этом режиме все результаты экспериментов сохраняются локально в структурированном формате, который можно загрузить в ClearML позже.

## Настройка

### 1. Создайте файл .env

Создайте файл `.env` в корне проекта со следующим содержимым:

```bash
# Режим работы: false = без ClearML, true = с ClearML
USE_CLEARML=false

# Настройки ClearML (используются только если USE_CLEARML=true)
CLEARML_API_HOST=http://51.250.43.3:8008
CLEARML_WEB_HOST=http://51.250.43.3:8080
CLEARML_FILES_HOST=http://51.250.43.3:8081
CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
CLEARML_S3_BUCKET=clearml-artifacts
CLEARML_S3_ACCESS_KEY=your_access_key
CLEARML_S3_SECRET_KEY=your_secret_key
CLEARML_S3_REGION=us-east-1
CLEARML_S3_PATH_STYLE=true
CLEARML_S3_VERIFY_SSL=false
```

### 2. Запуск экспериментов

Эксперименты автоматически определяют режим работы из `.env` файла:

```bash
# Автоматически использует USE_CLEARML из .env
poetry run python run_experiment_simple.py

# Принудительно отключить ClearML (игнорирует .env)
poetry run python run_experiment_simple.py --no-clearml

# Принудительно включить ClearML (игнорирует .env)
poetry run python run_experiment_simple.py --use-clearml
```

## Структура сохраненных результатов

При работе без ClearML результаты сохраняются в следующей структуре:

```
outputs/
└── experiment_name/
    ├── metadata.json              # Метаданные эксперимента
    ├── results.json                # Основные результаты (метрики)
    ├── predictions.json           # Предсказания модели
    ├── memory_usage.json          # Использование памяти
    ├── config/
    │   └── experiment_config.json # Полная конфигурация эксперимента
    ├── logs/
    │   └── text_logs.json         # Текстовые логи
    ├── metrics/
    │   ├── scalar_metrics.json    # Скалярные метрики (графики)
    │   ├── single_values.json     # Одиночные значения
    │   ├── tables.json            # Таблицы метрик
    │   └── *.csv                  # CSV файлы для каждой метрики
    └── artifacts/
        ├── artifacts_metadata.json # Метаданные артефактов
        ├── experiment_results_*.json
        ├── model_predictions_*.json
        └── memory_usage_*.json
```

## Загрузка результатов в ClearML

После завершения эксперимента в режиме без ClearML, вы можете загрузить результаты в ClearML:

```bash
# Базовая загрузка (название задачи берется из metadata.json)
poetry run python upload_results_to_clearml.py outputs/experiment_name/

# С указанием проекта и названия задачи
poetry run python upload_results_to_clearml.py outputs/experiment_name/ \
    --project slm-experiments \
    --task-name my_experiment \
    --tags model_name dataset_name

# С указанием .env файла
poetry run python upload_results_to_clearml.py outputs/experiment_name/ \
    --env-file .env
```

## Формат данных

Все данные сохраняются в JSON формате с метаданными:

- **metadata.json**: Название эксперимента, временная метка, версия формата
- **experiment_config.json**: Полная конфигурация (модель, датасет, параметры)
- **text_logs.json**: Массив текстовых логов с временными метками
- **scalar_metrics.json**: Скалярные метрики с итерациями (для графиков)
- **single_values.json**: Одиночные значения метрик
- **tables.json**: Таблицы метрик
- **artifacts_metadata.json**: Метаданные артефактов (файлы результатов)

## Преимущества режима без ClearML

1. ✅ Работа без доступа к серверу ClearML и MinIO
2. ✅ Все данные сохраняются локально
3. ✅ Структурированный формат для анализа
4. ✅ Возможность загрузки в ClearML позже
5. ✅ CSV файлы для удобного анализа метрик

## Примеры использования

### Запуск эксперимента без ClearML

```bash
# Убедитесь, что USE_CLEARML=false в .env
poetry run python run_experiment_simple.py
```

### Просмотр результатов

```bash
# Просмотр основных метрик
cat outputs/experiment_name/results.json | jq

# Просмотр метрик в CSV формате
cat outputs/experiment_name/metrics/*.csv

# Просмотр логов
cat outputs/experiment_name/logs/text_logs.json | jq
```

### Загрузка в ClearML

```bash
# После настройки доступа к ClearML
poetry run python upload_results_to_clearml.py outputs/experiment_name/
```

