# Фреймворк для экспериментов с малыми языковыми моделями

Этот фреймворк предоставляет гибкую систему для проведения экспериментов с малыми языковыми моделями (smolLM2, Qwen3) и различными ретриверами.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Установка основных зависимостей
poetry install

# Или через pip в виртуальном окружении Poetry
poetry run pip install clearml omegaconf hydra-core pandas
```

### 2. Тест подключения к S3
```bash
poetry run python test_s3_connection.py --bucket datasets
```

### 3. Получение данных
```bash
# Быстрый способ - скачивание готовых данных (рекомендуется!)
poetry run python download_processed_data.py --mode both --bucket datasets

# Или полная обработка данных (медленнее)
poetry run python process_s3_data.py --mode both --bucket datasets --no-upload

# После обработки данные будут в папке data/
ls -la data/nq/
ls -la data/simple_qa/
```

### 4. Запуск эксперимента
```bash
# Полный эксперимент на всех данных (~3610 примеров)
# Конфигурация берется из configs/config.yaml
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py

# Эксперимент без ClearML логирования
poetry run python run_experiment_simple.py --no-clearml

# Для запуска с другими параметрами - отредактируйте configs/config.yaml:
# - dataset: local_nq или local_simple_qa
# - model: smollm2_135m, smollm2_360m, smollm2_1.7b
# - experiment_mode: no_context, test_10_samples, test_100_samples
```

## 📊 Новые возможности

- **📊 ClearML интеграция** - полное логирование экспериментов в ClearML
- **🎯 GPU поддержка** - автоматическое использование Tesla V100 32GB
- **⚡ Быстрое скачивание готовых данных** - готовые обработанные датасеты из S3
- **🔄 Автоматическая обработка данных из S3** - скачивание, конвертация и загрузка результатов
- **📥 Скачивание данных из S3** - удобные скрипты для работы с данными
- **📤 Загрузка данных в S3** - синхронизация локальных и облачных данных
- **🔍 Тестирование S3** - проверка подключения и доступности данных
- **📁 Управление данными** - полный цикл работы с данными в облаке
- **💾 Отслеживание памяти** - мониторинг CPU и GPU памяти

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
- NVIDIA GPU с CUDA (опционально, но рекомендуется для ускорения)
  - Tesla V100 32GB или аналогичная
  - NVIDIA драйверы 550+ для поддержки CUDA

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

**Новые зависимости:**
- `boto3` - для работы с AWS S3
- `python-dotenv` - для загрузки переменных окружения из .env файла
- `clearml` - для логирования экспериментов
- `omegaconf` и `hydra-core` - для управления конфигурациями
- `pandas` - для создания таблиц результатов

3. **Активация виртуального окружения:**
```bash
# Активация shell
poetry shell

# Или запуск команд через poetry run
poetry run python script.py
```

4. **Настройка переменных окружения:**
```bash
# Файл .env уже содержит настройки S3 для проекта
# Проверьте, что файл .env содержит корректные креденшиалы:
cat .env | grep CLEARML_S3

# Если нужно изменить настройки S3, отредактируйте .env:
# CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
# CLEARML_S3_BUCKET=clearml-artifacts
# CLEARML_S3_ACCESS_KEY=your_key
# CLEARML_S3_SECRET_KEY=your_secret
# CLEARML_S3_REGION=us-east-1
```

5. **Проверка GPU (опционально, но рекомендуется):**
```bash
# Проверка наличия NVIDIA GPU
nvidia-smi

# Проверка доступности CUDA в PyTorch
poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Если GPU не определяется, возможно нужно установить/обновить драйвера NVIDIA:
# sudo apt update
# sudo apt install nvidia-driver-550  # для Tesla V100
# sudo reboot  # перезагрузка требуется после установки драйверов
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
   # Тест NQ данных
   poetry run python test_local_data.py
   
   # Тест SimpleQA данных
   poetry run python test_simple_qa_data.py
   ```

### S3 данные (опционально)

Проект поддерживает работу с данными в S3 хранилище. В bucket `datasets` уже хранятся исходные файлы:
- `NQ-open.dev.merged.jsonl` - исходный файл Natural Questions
- `simple_qa_test_set_with_documents.csv` - исходный файл SimpleQA

#### 🔧 Настройка S3

1. **Проверка подключения к S3**:
   ```bash
   # Тест подключения к S3
   poetry run python test_s3_connection.py --bucket datasets
   
   # Проверка наличия файлов датасетов
   poetry run python test_s3_connection.py --bucket datasets --check-datasets
   ```

#### 📊 Обработка данных из S3

**Рекомендуемый способ** - автоматическая обработка данных:

```bash
# Обработка NQ данных (скачать исходный файл, конвертировать, загрузить результат)
poetry run python process_s3_data.py --mode nq --bucket datasets

# Обработка SimpleQA данных
poetry run python process_s3_data.py --mode simple_qa --bucket datasets

# Обработка всех данных сразу
poetry run python process_s3_data.py --mode both --bucket datasets

# Обработка без загрузки результатов в S3 (только локально)
# Обрабатывает оба датасета: NQ и SimpleQA
poetry run python process_s3_data.py --mode both --bucket datasets --no-upload
```

#### 📥 Скачивание данных из S3

**Быстрый способ - скачивание готовых обработанных данных:**
```bash
# Скачивание готовых NQ данных (быстро!)
poetry run python download_processed_data.py --mode nq --bucket datasets

# Скачивание готовых SimpleQA данных (быстро!)
poetry run python download_processed_data.py --mode simple_qa --bucket datasets

# Скачивание всех готовых данных (рекомендуется!)
poetry run python download_processed_data.py --mode both --bucket datasets
```

**Полная обработка данных (медленнее, но создает все файлы):**
```bash
# Скачивание NQ данных (сохраняется в data/nq/)
poetry run python download_from_s3.py --mode nq --bucket datasets

# Скачивание SimpleQA данных (сохраняется в data/simple_qa/)
poetry run python download_from_s3.py --mode simple_qa --bucket datasets

# Просмотр содержимого S3 bucket
poetry run python download_from_s3.py --mode browse --bucket datasets

# Скачивание произвольного файла
poetry run python download_from_s3.py --mode custom \
  --s3-key NQ-open.dev.merged.jsonl \
  --local-file data/nq/NQ-open.dev.merged.jsonl \
  --bucket datasets
```

#### 📤 Загрузка данных в S3

```bash
# Загрузка NQ данных (из data/nq/)
poetry run python upload_to_s3.py --mode nq --bucket datasets

# Загрузка SimpleQA данных (из data/simple_qa/)
poetry run python upload_to_s3.py --mode simple_qa --bucket datasets

# Загрузка произвольного файла
poetry run python upload_to_s3.py --mode custom \
  --local-file data/my_data.json \
  --s3-key my_data.json \
  --bucket datasets
```

#### 🔍 Тестирование S3

```bash
# Базовый тест подключения
poetry run python test_s3_connection.py --bucket datasets

# Тест с проверкой операций с файлами
poetry run python test_s3_connection.py --bucket datasets --test-operations

# Тест с проверкой файлов датасетов
poetry run python test_s3_connection.py --bucket datasets --check-datasets

# Полный тест
poetry run python test_s3_connection.py --bucket datasets --test-operations --check-datasets
```

#### 📁 Структура данных в S3

```
s3://datasets/
├── NQ-open.dev.merged.jsonl                    # Исходный файл NQ
├── simple_qa_test_set_with_documents.csv        # Исходный файл SimpleQA
├── nq_full_dataset.json                         # Конвертированный NQ (после обработки)
├── nq_converted_eval.json                       # NQ eval данные (после обработки)
├── nq_converted_train.json                      # NQ train данные (после обработки)
├── simple_qa_converted.json                     # Конвертированный SimpleQA (после обработки)
├── simple_qa_train.json                         # SimpleQA train данные (после обработки)
└── simple_qa_eval.json                          # SimpleQA eval данные (после обработки)
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
# Базовый запуск с конфигурацией из configs/config.yaml
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py

# Запуск без ClearML
poetry run python run_experiment_simple.py --no-clearml

# Для изменения параметров (модель, датасет, режим) - 
# отредактируйте файл configs/config.yaml
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
# Запуск в фоне с логированием (с ClearML)
# Конфигурация берется из configs/config.yaml
CLEARML_CONFIG_FILE=./clearml.conf nohup poetry run python run_experiment_simple.py > experiment.log 2>&1 &

# Или без ClearML
nohup poetry run python run_experiment_simple.py --no-clearml > experiment.log 2>&1 &

# Мониторинг прогресса
tail -f experiment.log

# Проверка процесса
ps aux | grep python

# Мониторинг GPU
watch -n 1 nvidia-smi
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
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py
# Ctrl+A, D для отключения от screen

# Возврат к screen
screen -r experiment

# Остановка: Ctrl+C в screen

# Или через tmux
tmux new-session -s experiment
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py
# Ctrl+B, D для отключения

# Возврат к tmux
tmux attach -t experiment
```

#### 📊 **Доступные опции (указываются в configs/config.yaml):**

**Модели** (в defaults: model):
- `smollm2_135m` - SmolLM-135M (быстрая, 135M параметров)
- `smollm2_360m` - SmolLM-360M (средняя, 360M параметров)
- `smollm2_1.7b` - SmolLM-1.7B (большая, 1.7B параметров)
- `qwen_0.6b` - Qwen-0.6B
- `qwen_1.7b` - Qwen-1.7B
- `qwen_4b` - Qwen-4B

**Датасеты** (в defaults: dataset):
- `local_nq` - Natural Questions (локально, 3610 примеров)
- `local_simple_qa` - SimpleQA (локально)
- `rag_nq` - Natural Questions (для RAG экспериментов)

**Режимы экспериментов** (в defaults: experiment_mode):
- `no_context` - без контекста, все примеры
- `test_10_samples` - тестовый режим, 10 примеров
- `test_100_samples` - тестовый режим, 100 примеров
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
   - Отслеживание CPU и GPU памяти (логируется автоматически в ClearML)
   - Мониторинг прогресса: `tail -f experiment.log`
   - Проверка процессов: `ps aux | grep python`
   - GPU мониторинг в реальном времени: `watch -n 1 nvidia-smi`
   - Проверка CUDA в PyTorch: `poetry run python -c "import torch; print('CUDA:', torch.cuda.is_available())"`
   - Вся статистика памяти сохраняется в `outputs/<experiment_name>/memory_usage.json`

3. **Отладка проблем**:
   - Проверьте логи в `outputs/<experiment_name>/`
   - При OOM ошибках уменьшите batch_size или включите оптимизации памяти
   - При проблемах с данными проверьте кэш и S3 доступ

### Сравнение экспериментов

1. **Запуск серии экспериментов**:
   ```bash
   # Для запуска нескольких экспериментов с разными параметрами:
   # 1. Отредактируйте configs/config.yaml, измените модель на smollm2_135m
   # 2. Запустите: CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py
   # 3. Отредактируйте configs/config.yaml, измените модель на smollm2_360m
   # 4. Запустите: CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py
   # И т.д.
   
   # Все результаты будут сохранены в ClearML и доступны для сравнения
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
│   ├── nq/                 # Natural Questions
│   │   ├── NQ-open.dev.merged.jsonl      # Исходный файл
│   │   └── nq_full_dataset.json          # Полный датасет (3610 примеров)
│   └── simple_qa/          # SimpleQA
│       ├── simple_qa_test_set_with_documents.csv  # Исходный файл
│       └── simple_qa_converted.json     # Конвертированный датасет
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
├── .env                   # Переменные окружения (S3 креденшиалы)
├── run_experiment_simple.py  # Основной скрипт запуска
├── convert_nq_data.py     # Скрипт конвертации NQ данных
├── convert_simple_qa_data.py  # Скрипт конвертации SimpleQA данных
├── test_local_data.py     # Тест загрузки локальных NQ данных
├── test_simple_qa_data.py  # Тест загрузки локальных SimpleQA данных
├── upload_to_s3.py        # Загрузка данных в S3
├── download_from_s3.py    # Скачивание данных из S3
├── download_processed_data.py  # Скачивание готовых обработанных данных (быстро!)
├── process_s3_data.py     # Обработка данных из S3 (автоматическая)
├── test_s3_connection.py  # Тестирование подключения к S3
└── S3_DATA_MANAGEMENT.md  # Документация по работе с S3
└── CLEARML_INTEGRATION.md  # Документация по интеграции с ClearML
```

## 📊 Логирование в ClearML

Фреймворк поддерживает полное логирование экспериментов в ClearML:

### Что логируется:
- **Полная конфигурация эксперимента** - все параметры модели, датасета, ретривера
- **Метрики в реальном времени** - Token Recall, время выполнения, использование памяти
- **Предсказания модели** - все вопросы, ответы и контексты для анализа
- **Артефакты** - файлы результатов, предсказаний и статистики памяти

### Быстрый старт с ClearML:
```bash
# Запуск эксперимента с ClearML (конфигурация из configs/config.yaml)
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py

# Запуск без ClearML
poetry run python run_experiment_simple.py --no-clearml

# Для изменения параметров (количество сэмплов, модель, датасет):
# Отредактируйте configs/config.yaml, измените:
# - experiment_mode: no_context (все данные) / test_10_samples / test_100_samples
# - model: smollm2_135m / smollm2_360m / smollm2_1.7b
# - dataset: local_nq / local_simple_qa
```

### Настройка ClearML:
Убедитесь, что файл `.env` содержит корректные настройки ClearML:
```bash
# Проверка настроек
cat .env | grep CLEARML
```

### Хранение результатов:
- **ClearML Database** - метрики, конфигурации, метаданные (легковесные данные)
- **MinIO S3** - артефакты (predictions.json, results.json, memory_usage.json - тяжелые данные)
- Артефакты хранятся в бакете `s3://51.250.43.3:9000/clearml-artifacts`
- Проверка артефактов: `poetry run python check_minio_artifacts.py`

### Доступные метрики:
- **Scalars** - итоговые числовые значения (Token Recall, Total Time, и др.)
- **Plots** - таблица с итоговыми метриками и графики прогресса
- **Console** - полный лог эксперимента
- **Debug Samples** - примеры предсказаний модели (первые 100)

Подробная документация: [CLEARML_INTEGRATION.md](CLEARML_INTEGRATION.md)
