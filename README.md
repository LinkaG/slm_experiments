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
   - Полный датасет: `data/nq/nq_full_dataset.json` (3610 примеров)
   - Исходный файл: `data/nq/NQ-open.dev.merged.jsonl`
   - Конвертированный для RAG тестирования

2. **Конвертация данных**:
   ```bash
   # Конвертация полного датасета
   poetry run python convert_nq_data.py \
     --input data/nq/NQ-open.dev.merged.jsonl \
     --output data/nq/nq_full_dataset.json
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
   poetry run python run_experiment_simple.py experiment_mode=no_context
   ```

2. **С идеальным контекстом** (oracle_context):
   - Использование ground truth контекста из датасета
   - Оценка верхней границы производительности модели
   ```bash
   poetry run python run_experiment_simple.py experiment_mode=oracle_context
   ```

3. **С ретривером** (retriever_context):
   - Стандартный режим с использованием ретривера
   - Оценка производительности всего пайплайна
   ```bash
   poetry run python run_experiment_simple.py experiment_mode=retriever_context
   ```

### Запуск экспериментов

#### 🚀 **Основной способ (рекомендуемый):**

```bash
# Базовый запуск с текущей конфигурацией
poetry run python run_experiment_simple.py

# Изменение модели
poetry run python run_experiment_simple.py model=qwen_1.7b

# Изменение датасета
poetry run python run_experiment_simple.py dataset=local_simple_qa

# Изменение режима
poetry run python run_experiment_simple.py experiment_mode=no_context

# Комбинированные изменения
poetry run python run_experiment_simple.py model=qwen_1.7b dataset=local_nq experiment_mode=no_context
```

#### 🔧 **Настройка через config.yaml:**

Отредактируйте `configs/config.yaml`:

```yaml
defaults:
  - model: smollm2_135m        # Модель по умолчанию
  - dataset: rag_nq           # Датасет по умолчанию  
  - experiment_mode: no_context # Режим по умолчанию
  - _self_

experiment:
  name: ${model.name}_${dataset.name}_${experiment_mode.name}
  output_dir: outputs/${experiment.name}
  seed: 42
  log_predictions: true
  save_contexts: true
  max_samples: null  # Использовать все данные
```

#### 🌐 **Запуск в фоновом режиме:**

```bash
# Запуск в фоне с логированием
nohup poetry run python run_experiment_simple.py > experiment.log 2>&1 &

# Мониторинг прогресса
tail -f experiment.log

# Проверка процесса
ps aux | grep python
```

#### 🛑 **Остановка фонового эксперимента:**

```bash
# Остановить все процессы по имени скрипта
pkill -f "run_experiment_simple.py"

# Принудительная остановка (если не помогает)
pkill -9 -f "run_experiment_simple.py"

# Остановить по PID (если знаете номер процесса)
kill <PID>

# Проверить, что все остановлено
ps aux | grep run_experiment_simple | grep -v grep
```

#### 🖥️ **Альтернатива: screen/tmux (рекомендуется):**

```bash
# Запуск в screen
screen -S experiment
poetry run python run_experiment_simple.py
# Ctrl+A, D для отключения от screen

# Возврат к screen
screen -r experiment

# Остановка: Ctrl+C в screen

# Или через tmux
tmux new-session -s experiment
poetry run python run_experiment_simple.py
# Ctrl+B, D для отключения

# Возврат к tmux
tmux attach -t experiment
```

#### 📊 **Доступные опции:**

**Модели:**
- `smollm2_135m` - SmolLM-135M
- `smollm2_360m` - SmolLM-360M  
- `smollm2_1.7b` - SmolLM-1.7B
- `qwen_0.6b` - Qwen-0.6B
- `qwen_1.7b` - Qwen-1.7B
- `qwen_4b` - Qwen-4B

**Датасеты:**
- `rag_nq` - Natural Questions (полный датасет)
- `local_nq` - Natural Questions (локально)
- `local_simple_qa` - SimpleQA (локально)

**Режимы:**
- `no_context` - без контекста
- `oracle_context` - с оракульным контекстом
- `retriever_context` - с ретривером

### Мониторинг и анализ

1. **Отслеживание прогресса**:
   - Логи сохраняются в `experiment.log`
   - Результаты в `outputs/<experiment_name>/results.json`
   - Мониторинг в реальном времени: `tail -f experiment.log`

2. **Анализ результатов**:
   - Основные метрики: Token Recall, время выполнения
   - Детальные результаты в JSON формате
   - Предсказания модели сохраняются для анализа
   - Token Recall показывает качество ответов модели
   - Детальная статистика использования памяти в `outputs/<experiment_name>/memory_usage.json`

3. **Мониторинг ресурсов**:
   - Отслеживание CPU и GPU памяти
   - Мониторинг прогресса: `tail -f experiment.log`
   - Проверка процессов: `ps aux | grep python`
   - GPU мониторинг: `nvidia-smi`

3. **Отладка проблем**:
   - Проверьте логи в `outputs/<experiment_name>/`
   - При OOM ошибках уменьшите batch_size или включите оптимизации памяти
   - При проблемах с данными проверьте кэш и S3 доступ

### Сравнение экспериментов

1. **Запуск серии экспериментов**:
   ```bash
   # Сравнение разных размеров модели
   poetry run python run_experiment_simple.py model=smollm2_135m
   poetry run python run_experiment_simple.py model=smollm2_360m
   poetry run python run_experiment_simple.py model=smollm2_1.7b
   
   # Сравнение разных датасетов
   poetry run python run_experiment_simple.py dataset=rag_nq
   poetry run python run_experiment_simple.py dataset=local_simple_qa
   ```

2. **Анализ результатов**:
   - Сравнивайте Token Recall между экспериментами
   - Анализируйте время выполнения для разных моделей
   - Учитывайте использование памяти при выборе модели

### Добавление новых конфигураций

1. **Создание конфига модели**:
   - Скопируйте существующий конфиг из `configs/model/`
   - Измените параметры под ваши нужды
   - Сохраните с новым именем

2. **Настройка эксперимента**:
   - При необходимости добавьте новые параметры в `configs/config.yaml`
   - Создайте новые конфиги датасетов в `configs/dataset/`
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

### Добавление нового ретривера (для будущего использования)
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
│       ├── NQ-open.dev.merged.jsonl      # Исходный файл
│       └── nq_full_dataset.json          # Полный датасет (3610 примеров)
├── configs/                # Конфигурации Hydra
│   ├── model/             # Конфигурации моделей
│   ├── dataset/           # Конфигурации датасетов
│   └── experiment_mode/   # Режимы экспериментов
├── src/                   # Исходный код
│   ├── data/              # Обработка датасетов
│   ├── models/            # Реализации моделей
│   ├── retrievers/        # Реализации ретриверов
│   ├── experiment/        # Запуск экспериментов и метрики
│   └── utils/             # Общие утилиты
├── outputs/               # Результаты экспериментов
├── .cache/                # Кэш данных
├── pyproject.toml         # Poetry конфигурация
├── run_experiment_simple.py  # Основной скрипт запуска
├── convert_nq_data.py     # Скрипт конвертации данных
└── test_local_data.py     # Тест загрузки данных
```
