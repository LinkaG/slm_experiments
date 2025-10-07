# Настройка логирования в ClearML

## Обзор

Проект `slm_experiments` настроен для работы с удаленным сервером ClearML. Логирование включает:

- **Метрики экспериментов** (токен-реколл, время выполнения)
- **Использование памяти** (CPU/GPU)
- **Артефакты** (результаты, предсказания, логи)
- **Конфигурации** (параметры моделей, датасетов, ретриверов)

## Конфигурация сервера

### Адреса сервера
- **ClearML Web UI**: http://51.250.43.3:8080
- **ClearML API**: http://51.250.43.3:8008
- **ClearML Files**: http://51.250.43.3:8081
- **MinIO S3**: http://51.250.43.3:9000

### Учетные данные S3
- **Access Key**: `minio_admin_2024`
- **Secret Key**: `Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0`
- **Bucket**: `clearml-artifacts`
- **Region**: `us-east-1`

## Настройка проекта

### 1. Конфигурационные файлы

#### ~/.clearml.conf
```hocon
api {
    api_server: http://51.250.43.3:8008
    web_server: http://51.250.43.3:8080
    files_server: http://51.250.43.3:8081
    credentials {
        "access_key" = "your-access-key"
        "secret_key" = "your-secret-key"
    }
}
sdk {
    storage {
        cache {
            default_base_dir: "~/clearml/cache"
        }
        s3 {
            endpoint_url: "http://51.250.43.3:9000"
            bucket: "clearml-artifacts"
            access_key: "minio_admin_2024"
            secret_key: "Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0"
            region: "us-east-1"
            path_style: true
            verify_ssl: false
        }
    }
}
```

#### .env
```bash
CLEARML_API_HOST=http://51.250.43.3:8008
CLEARML_WEB_HOST=http://51.250.43.3:8080
CLEARML_FILES_HOST=http://51.250.43.3:8081
CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
CLEARML_S3_BUCKET=clearml-artifacts
CLEARML_S3_ACCESS_KEY=minio_admin_2024
CLEARML_S3_SECRET_KEY=Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0
```

### 2. Получение учетных данных

1. Откройте веб-интерфейс: http://51.250.43.3:8080
2. Зарегистрируйтесь или войдите в систему
3. Перейдите в **Settings → Workspace → Create new credentials**
4. Скопируйте `access_key` и `secret_key`
5. Обновите файл `~/.clearml.conf`

## Логирование в коде

### Основные компоненты

#### 1. ExperimentRunner (`src/experiment/runner.py`)

```python
# Инициализация ClearML
self.task = Task.init(
    project_name="slm-experiments",
    task_name=self.config.name,
    auto_connect_frameworks=False
)

# Логирование конфигурации
self.task.connect({
    "model": self.config.model_config,
    "retriever": self.config.retriever_config,
    "dataset": self.config.dataset_config,
    "metrics": self.config.metrics_config
})

# Логирование метрик
self.logger.report_scalar(
    title="metrics",
    series="token_recall",
    value=recall,
    iteration=0
)

# Загрузка артефактов
self.task.upload_artifact(
    name="experiment_results",
    artifact_object=results_file,
    metadata={...}
)
```

#### 2. MemoryTracker (`src/utils/memory_tracker.py`)

```python
# Логирование использования памяти
logger.report_scalar(
    title=f"memory/{component}",
    series=key,
    value=value,
    iteration=len(self.memory_log)
)
```

#### 3. PredictionsTracker (`src/utils/predictions_tracker.py`)

```python
# Загрузка предсказаний как артефакт
task.upload_artifact(
    name="predictions",
    artifact_object=predictions_file,
    metadata={
        "num_predictions": len(self.predictions),
        "context_types": self._get_unique_context_types(),
        "model_names": self._get_unique_model_names()
    }
)
```

### Логируемые метрики

#### Основные метрики
- **Token Recall**: качество ответов модели
- **Время выполнения**: длительность эксперимента
- **Количество примеров**: размер датасета

#### Метрики памяти
- **CPU Memory**: использование оперативной памяти
- **GPU Memory**: использование видеопамяти
- **Peak Usage**: пиковое использование

#### Метрики модели
- **Размер модели**: количество параметров
- **Размер индекса ретривера**: количество документов
- **Статистика датасета**: размер, количество примеров

### Артефакты

#### 1. Результаты эксперимента
- **Файл**: `outputs/<experiment_name>/results.json`
- **Содержимое**: финальные метрики
- **Метаданные**: конфигурация эксперимента

#### 2. Предсказания модели
- **Файл**: `outputs/<experiment_name>/predictions.json`
- **Содержимое**: все предсказания с контекстами
- **Метаданные**: статистика по типам контекста

#### 3. Логи памяти
- **Файл**: `outputs/<experiment_name>/memory_usage.json`
- **Содержимое**: детальная статистика использования памяти
- **Метаданные**: пиковое использование

## Запуск экспериментов

### Базовый запуск
```bash
poetry run python -m src.cli run-experiment \
    model=smollm2_135m \
    dataset=local_nq
```

### С полной конфигурацией
```bash
poetry run python -m src.cli run-experiment \
    model=smollm2_135m \
    dataset=local_nq \
    retriever=bm25 \
    experiment.name=my_custom_experiment
```

### Тестирование подключения
```bash
poetry run python test_clearml_connection.py
```

## Мониторинг результатов

### Веб-интерфейс
1. Откройте http://51.250.43.3:8080
2. Найдите ваш эксперимент по имени
3. Просматривайте метрики в реальном времени
4. Скачивайте артефакты

### API доступ
```python
from clearml import Task

# Получение задачи
task = Task.get_task(task_id="your-task-id")

# Получение метрик
metrics = task.get_last_scalar_metrics()

# Получение артефактов
artifacts = task.artifacts
```

## Устранение проблем

### Проблема: Сервер недоступен
```bash
# Проверка доступности
ping 51.250.43.3
curl http://51.250.43.3:8080
```

### Проблема: Ошибки аутентификации
1. Проверьте учетные данные в `~/.clearml.conf`
2. Убедитесь, что сервер доступен
3. Проверьте настройки S3

### Проблема: Ошибки загрузки артефактов
1. Проверьте настройки S3 в конфигурации
2. Убедитесь в доступности MinIO
3. Проверьте права доступа к bucket

## Дополнительные возможности

### Сравнение экспериментов
- Используйте веб-интерфейс для сравнения метрик
- Экспортируйте данные для анализа
- Создавайте дашборды с ключевыми метриками

### Автоматизация
- Настройте уведомления о завершении экспериментов
- Используйте API для автоматического анализа результатов
- Интегрируйте с системами CI/CD

### Масштабирование
- Настройте кластер для распределенных экспериментов
- Используйте очереди задач для планирования
- Настройте мониторинг ресурсов
