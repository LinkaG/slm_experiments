"""Датасет в едином формате qa_pairs.jsonl для режима no_context."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

from .base import BaseDataset, DatasetItem


class QAPairsJsonlDataset(BaseDataset):
    """
    Файл: qa_pairs.jsonl, по строке на объект:
    {"question_id": ..., "question": "...", "answer": ["...", ...]}

    Используется только оценка через get_eval_data().
    """

    def __init__(self, dataset_config: Dict[str, Any]):
        self.config = dataset_config
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(dataset_config.get("cache_dir", ".cache/datasets"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        path = Path(dataset_config["qa_pairs_path"])
        self._records = self._load(path)

    def _load(self, path: Path) -> List[dict]:
        cache_name = f"{self.config.get('name', path.parent.name)}_{path.stem}.json"
        cache_file = self.cache_dir / cache_name
        if cache_file.exists():
            self.logger.info(f"Loading qa_pairs from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if not path.exists():
            raise FileNotFoundError(f"qa_pairs file not found: {path}")

        self.logger.info(f"Loading qa_pairs from: {path}")
        records: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        return records

    @staticmethod
    def _normalize_answer(raw: Any) -> Union[List[str], None]:
        if raw is None:
            return None
        if isinstance(raw, list):
            return [str(x) for x in raw if x is not None and str(x).strip()]
        s = str(raw).strip()
        return [s] if s else None

    def _iter_records(self, records: List[dict]) -> Iterator[DatasetItem]:
        for item in records:
            qid = item.get("question_id")
            question = item.get("question")
            if not question:
                continue
            answers = self._normalize_answer(item.get("answer"))
            yield DatasetItem(
                question=question,
                answer=answers,
                context=None,
                metadata={
                    "id": qid,
                    "question_id": qid,
                    "all_answers": answers,
                },
            )

    def get_eval_data(self) -> Iterator[DatasetItem]:
        return self._iter_records(self._records)

    def get_dataset_stats(self) -> Dict[str, Any]:
        return {"eval_size": len(self._records)}
