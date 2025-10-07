# Настройка окружения для slm_experiments

## Установка Poetry

Poetry уже установлен в системе. Если нужно установить заново:

```bash
pip3 install poetry
```

## Настройка проекта

1. **Активация Poetry окружения:**
```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/dolganov/slm_experiments
```

2. **Установка зависимостей:**
```bash
poetry install
```

3. **Активация виртуального окружения:**
```bash
poetry shell
```

## Работа с данными

### Конвертация Natural Questions

Ваш датасет уже конвертирован и находится в:
- `data/nq/nq_converted_train.json` (80 примеров)
- `data/nq/nq_converted_eval.json` (20 примеров)

### Конвертация дополнительных данных

Для конвертации большего количества данных:

```bash
# Конвертация с ограничением (100 примеров)
poetry run python convert_nq_data.py \
  --input data/nq/NQ-open.dev.merged.jsonl \
  --output data/nq/nq_converted.json \
  --max-items 1000 \
  --split

# Конвертация всего датасета
poetry run python convert_nq_data.py \
  --input data/nq/NQ-open.dev.merged.jsonl \
  --output data/nq/nq_converted.json \
  --split
```

### Тестирование загрузки данных

```bash
poetry run python test_local_data.py
```

## Запуск экспериментов

После настройки окружения можно запускать эксперименты:

```bash
# Базовый эксперимент с локальными данными
poetry run python -m src.cli run-experiment dataset=local_nq

# Эксперимент с конкретной моделью
poetry run python -m src.cli run-experiment \
  model=smollm2_135m \
  dataset=local_nq \
  experiment_mode=retriever_context
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
├── configs/                # Конфигурации
│   └── dataset/
│       └── local_nq.yaml   # Конфиг для локальных NQ данных
├── src/                    # Исходный код
├── pyproject.toml          # Poetry конфигурация
└── convert_nq_data.py     # Скрипт конвертации
```

## Полезные команды

```bash
# Просмотр зависимостей
poetry show

# Обновление зависимостей
poetry update

# Запуск в виртуальном окружении
poetry run python script.py

# Активация shell
poetry shell
```
