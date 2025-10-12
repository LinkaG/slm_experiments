# Шпаргалка по командам

## 🚀 Быстрый старт

### Запуск эксперимента
```bash
# С ClearML логированием (рекомендуется)
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py

# Без ClearML
poetry run python run_experiment_simple.py --no-clearml
```

### Запуск в фоне
```bash
# С ClearML
CLEARML_CONFIG_FILE=./clearml.conf nohup poetry run python run_experiment_simple.py > experiment.log 2>&1 &

# Без ClearML
nohup poetry run python run_experiment_simple.py --no-clearml > experiment.log 2>&1 &
```

## 🔧 Настройка эксперимента

**Все параметры задаются в `configs/config.yaml`:**

```yaml
defaults:
  - model: smollm2_135m        # Модель
  - dataset: local_nq          # Датасет  
  - experiment_mode: no_context # Режим
```

### Доступные опции:

**Модели:**
- `smollm2_135m` (135M параметров) ⚡ быстрая
- `smollm2_360m` (360M параметров)
- `smollm2_1.7b` (1.7B параметров)

**Датасеты:**
- `local_nq` - Natural Questions (3610 примеров)
- `local_simple_qa` - SimpleQA

**Режимы:**
- `no_context` - без контекста, все данные
- `test_10_samples` - тест на 10 примерах
- `test_100_samples` - тест на 100 примерах

## 📊 Мониторинг

```bash
# Просмотр логов
tail -f experiment.log

# Проверка процесса
ps aux | grep run_experiment_simple

# Мониторинг GPU
watch -n 1 nvidia-smi

# Проверка CUDA
poetry run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🛑 Остановка эксперимента

```bash
# Остановить процесс
pkill -f "run_experiment_simple.py"

# Принудительная остановка
pkill -9 -f "run_experiment_simple.py"
```

## 🌐 ClearML

**Просмотр результатов:**
- Web UI: http://51.250.43.3:8080
- Проект: slm_experiments
- Все метрики, графики и артефакты доступны в интерфейсе

**Хранение данных:**
- Метрики → ClearML Database
- Артефакты → MinIO S3 (s3://51.250.43.3:9000/clearml-artifacts)

## 📁 Результаты

Локально сохраняются в `outputs/<experiment_name>/`:
- `results.json` - итоговые метрики
- `predictions.json` - предсказания модели
- `memory_usage.json` - статистика памяти

## 🔄 Пример workflow

```bash
# 1. Редактируем конфигурацию
nano configs/config.yaml

# 2. Запускаем эксперимент в фоне
CLEARML_CONFIG_FILE=./clearml.conf nohup poetry run python run_experiment_simple.py > experiment.log 2>&1 &

# 3. Следим за прогрессом
tail -f experiment.log

# 4. Проверяем результаты в ClearML Web UI
# http://51.250.43.3:8080
```
