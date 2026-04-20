# Эксперименты с малыми языковыми моделями

Hydra-конфиги, один режим **`no_context`** (вопрос–ответ без внешнего контекста), датасеты в формате **`qa_pairs.jsonl`**. Основной сценарий на GPU: **`run_batch_experiments.py`** поднимает дочерние контейнеры Docker с монтированием репозитория и кэша моделей.

## Требования

- Python 3.10–3.13, [Poetry](https://python-poetry.org/)
- NVIDIA GPU и драйвер (для прогонов с CUDA)
- Для ClearML и MinIO: Docker-сеть **`clearml_backend`** (см. документацию вашего стенда ClearML)

## Установка

```bash
git clone <repository-url>
cd slm_experiments
poetry install
cp .env.example .env
# Заполните .env (ClearML, MinIO, HF_TOKEN при необходимости, DOCKER_MODELS_CACHE)
```

## Переменные окружения

См. **`.env.example`**. Важно для батча в Docker:

- **`DOCKER_MODELS_CACHE`** — каталог на хосте для Hugging Face / datasets (подкаталоги `huggingface/`, `datasets/`).
- **`CLEARML_S3_ACCESS_KEY` / `CLEARML_S3_SECRET_KEY` / `CLEARML_S3_REGION`** — подставляются в дочерние контейнеры из окружения хоста после `load_dotenv()` в батче.
- **`CLEARML_S3_ENDPOINT`** — обычно URL MinIO **с хоста** (IP или localhost).
- **`CLEARML_S3_DOCKER_ENDPOINT`** — URL MinIO **из контейнера**; по умолчанию в батче используется `http://minio:9000` (имя сервиса в сети Docker).

Конфиг ClearML внутри контейнера: **`clearml.conf.docker`** (копируется в `~/.clearml.conf` при старте SDK).

## Данные

Датасет задаётся в **`configs/dataset/*.yaml`**, тип **`qa_pairs_jsonl`**: файл **`qa_pairs.jsonl`** с полями `question_id`, `question`, `answer` (список строк).

## Одиночный эксперимент (на хосте)

```bash
poetry run python run_experiment_simple.py model=qwen_0.6b dataset=local_simple_qa experiment_mode=no_context
```

Без ClearML:

```bash
poetry run python run_experiment_simple.py --no-clearml model=qwen_0.6b dataset=local_simple_qa experiment_mode=no_context
```

По умолчанию см. **`configs/config.yaml`**.

## Docker-образ эксперимента

```bash
./build_docker_image.sh
```

Имя образа по умолчанию: **`slm-experiments:latest`**. Батч ожидает сеть **`clearml_backend`**.

## Пакетный запуск (рекомендуется)

С хоста (один процесс оркестратора; каждая задача — отдельный `docker run` с GPU):

```bash
# все модели × все датасеты из configs/model и configs/dataset
poetry run python run_batch_experiments.py --experiment-mode no_context

# выборочно
poetry run python run_batch_experiments.py \
  --experiment-mode no_context \
  --models qwen_0.6b \
  --datasets local_simple_qa

poetry run python run_batch_experiments.py --no-clearml --max-parallel 2
```

Лог батча: **`batch_experiments.log`**.

### Долгий прогон в tmux

```bash
tmux new-session -s slm-batch
cd /path/to/slm_experiments
poetry run python run_batch_experiments.py --experiment-mode no_context --models qwen_0.6b --datasets local_simple_qa
# отсоединиться: Ctrl+B, затем D
tmux attach -t slm-batch
tmux ls
# остановить сессию: внутри tmux Ctrl+C, затем exit; или: tmux kill-session -t slm-batch
```

## Результаты

Каталог: **`no_context/<имя_эксперимента>/`** (имя = `model_dataset_no_context` и т.п., см. Hydra).

Файлы:

- `<имя>_predictions.json`
- `<имя>_metrics.json`
- `<имя>_conf.json`

## Структура репозитория

```
slm_experiments/
├── configs/
│   ├── model/
│   ├── dataset/
│   └── experiment_mode/    # сейчас no_context
├── src/
│   ├── data/
│   ├── models/
│   ├── experiment/
│   └── utils/
├── run_experiment_simple.py
├── run_batch_experiments.py
├── build_docker_image.sh
├── Dockerfile.experiments
├── clearml.conf.docker
├── .env.example
└── pyproject.toml
```

## ClearML и MinIO

Метрики и задачи — в ClearML; тяжёлые артефакты загружаются в MinIO кодом эксперимента (boto3 с `endpoint_url` из `.env`). Для стабильной работы SDK в Docker в проекте отключена привязка `task.output_uri` к S3 через встроенную проверку ClearML; регистрация артефактов в задаче сохраняется.
