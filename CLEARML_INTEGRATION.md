# Интеграция с ClearML

Этот документ описывает интеграцию фреймворка с ClearML для логирования экспериментов.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
poetry install
```

### 2. Настройка ClearML
Убедитесь, что файл `.env` содержит корректные настройки ClearML:
```bash
cat .env | grep CLEARML
```

### 3. Тестирование интеграции
```bash
poetry run python test_clearml_integration.py
```

### 4. Запуск эксперимента с ClearML
```bash
# С ClearML логированием (по умолчанию)
poetry run python run_experiment_simple.py

# Без ClearML логирования
poetry run python run_experiment_simple.py --no-clearml

# С указанием файла .env
poetry run python run_experiment_simple.py --env-file /path/to/.env
```

## 📊 Что логируется в ClearML

### 1. Полная конфигурация эксперимента
- Настройки модели (название, размер, параметры)
- Настройки датасета (название, пути к файлам)
- Настройки ретривера (если используется)
- Параметры эксперимента (режим, количество примеров)

### 2. Метрики эксперимента
- **Token Recall** - основная метрика качества
- **Количество примеров** - количество обработанных примеров
- **Время выполнения** - общее время эксперимента
- **Использование памяти** - детальная статистика памяти

### 3. Предсказания модели
- Вопросы и ответы для всех примеров
- Ground truth ответы
- Контекст, использованный для генерации
- Token Recall для каждого примера
- Метаданные (ID примера, датасет, модель)

### 4. Артефакты
- **experiment_results.json** - финальные метрики
- **model_predictions.json** - все предсказания модели
- **memory_usage.json** - статистика использования памяти

## 🔧 Настройка

### Переменные окружения (.env файл)
```bash
# ClearML Configuration
CLEARML_API_HOST=http://51.250.43.3:8008
CLEARML_WEB_HOST=http://51.250.43.3:8080
CLEARML_FILES_HOST=http://51.250.43.3:8081

# S3 Storage Configuration (для артефактов)
CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
CLEARML_S3_BUCKET=clearml-artifacts
CLEARML_S3_ACCESS_KEY=your_access_key
CLEARML_S3_SECRET_KEY=your_secret_key
CLEARML_S3_REGION=us-east-1
CLEARML_S3_PATH_STYLE=true
CLEARML_S3_VERIFY_SSL=false
```

### Параметры командной строки
```bash
python run_experiment_simple.py [OPTIONS]

Options:
  --use-clearml          Использовать ClearML для логирования (по умолчанию: True)
  --no-clearml           Отключить ClearML логирование
  --env-file PATH        Путь к файлу .env с настройками ClearML
  --config-path PATH     Путь к конфигурационным файлам (по умолчанию: configs)
  --config-name NAME     Имя конфигурационного файла (по умолчанию: config)
```

## 📈 Мониторинг экспериментов

### 1. Веб-интерфейс ClearML
После запуска эксперимента вы получите ссылку на задачу в ClearML веб-интерфейсе.

### 2. Просмотр результатов
- **Scalars** - графики метрик в реальном времени
- **Text** - логи эксперимента и предсказания модели
- **Artifacts** - скачивание файлов результатов
- **Configuration** - полная конфигурация эксперимента

### 3. Сравнение экспериментов
- Используйте теги для группировки экспериментов
- Сравнивайте метрики между разными моделями
- Анализируйте предсказания для понимания ошибок

## 🛠️ Разработка

### Добавление новых метрик
```python
# В ExperimentRunner._evaluate()
metrics = {
    "token_recall": avg_recall,
    "num_examples": len(recalls),
    "new_metric": calculate_new_metric()  # Добавить новую метрику
}
```

### Кастомное логирование
```python
from src.utils.clearml_config import get_clearml_logger

logger = get_clearml_logger()
logger.report_scalar("custom", "metric_name", value, iteration)
logger.report_text("Custom log message")
```

### Загрузка артефактов
```python
# Загрузка результатов предыдущего эксперимента
task = Task.get_task(task_id="your_task_id")
results = task.artifacts["experiment_results"].get()
```

## 🔍 Отладка

### Проблемы с подключением
1. Проверьте настройки в `.env` файле
2. Убедитесь, что ClearML сервер доступен
3. Проверьте права доступа к S3

### Проблемы с логированием
1. Запустите тест интеграции: `python test_clearml_integration.py`
2. Проверьте логи в `experiment.log`
3. Убедитесь, что все зависимости установлены

### Режим без ClearML
Если ClearML недоступен, используйте флаг `--no-clearml`:
```bash
poetry run python run_experiment_simple.py --no-clearml
```

## 📚 Примеры использования

### Базовый эксперимент
```bash
# Эксперимент с ClearML логированием
poetry run python run_experiment_simple.py model=smollm2_1.7b dataset=local_nq

# Эксперимент без ClearML
poetry run python run_experiment_simple.py --no-clearml model=qwen_1.7b
```

### Серия экспериментов
```bash
# Сравнение разных моделей
for model in smollm2_135m smollm2_360m smollm2_1.7b; do
    poetry run python run_experiment_simple.py model=$model dataset=local_nq
done
```

### Фоновый запуск
```bash
# Запуск в фоне с логированием
nohup poetry run python run_experiment_simple.py > experiment.log 2>&1 &
```

## 🎯 Лучшие практики

1. **Именование экспериментов** - используйте описательные имена
2. **Теги** - добавляйте теги для группировки экспериментов
3. **Мониторинг** - следите за использованием памяти и времени
4. **Архивация** - регулярно архивируйте результаты экспериментов
5. **Документация** - документируйте изменения в конфигурации

## 🔗 Полезные ссылки

- [ClearML Documentation](https://clear.ml/docs)
- [ClearML Python SDK](https://clear.ml/docs/latest/docs/sdk/python_sdk)
- [ClearML Web Interface](http://51.250.43.3:8080)
