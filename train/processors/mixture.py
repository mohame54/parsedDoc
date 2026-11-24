"""Task mixing: draws one training example per document, weighted by task, with the wording, schema format
and output format re-drawn every epoch. Model-agnostic, so every transform uses it the same way.

    processor = DatasetProcessor(preset="structured")
    example = processor.sample(record, epoch=0, index=17)   # one example, task chosen by weight
"""
from __future__ import annotations

import random
from pathlib import Path

from ..derive import TASKS, derive
from ..describe import Describer
from .base import NoFittingExample


# Weights are relative. The "structured" preset is for the main goal: extract and produce suitable
# structured output schemas across formats; "balanced" keeps every task in play.
PRESETS: dict[str, dict[str, float]] = {
    "structured": {"extract_to_schema": 3, "propose_schema": 3, "extract": 4, "schema": 1,
                   "field_qa": 1, "table": 1, "visual": 1, "figure": 1, "chart_qa": 1, "chart_trend": 1, "ocr": 1,
                   "classify": 0.5, "key_value": 0.5},
    # for --train-vision: the visual and reasoning tasks are upweighted, the extraction core stays so it does not regress
    "visual": {"extract_to_schema": 3, "extract": 3, "propose_schema": 1, "visual": 2, "table": 2, "figure": 2,
               "chart_qa": 2, "chart_trend": 2, "checkbox_qa": 2, "table_qa": 2, "field_qa": 1, "ocr": 1},
    "balanced": {name: 1.0 for name in TASKS},
    "ocr": {"ocr": 1.0},
}


class DatasetProcessor:
    def __init__(self, tasks: dict[str, float] | list[str] | None = None, preset: str = "structured",
                 descriptions: "str | Path | dict[str, str] | None" = None):
        """`descriptions`: a JSON file (or dict) of attribute descriptions from `labeller describe-*`; fields
        without an entry get a template description."""
        self.describer = Describer(descriptions) if isinstance(descriptions, dict) else Describer.load(descriptions)
        if tasks is None:
            tasks = PRESETS[preset]
        elif not isinstance(tasks, dict):
            tasks = {name: 1.0 for name in tasks}
        unknown = set(tasks) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown tasks {sorted(unknown)}; available: {sorted(TASKS)}")
        self.weights = {name: float(w) for name, w in tasks.items() if w > 0}

    def expand(self, record: dict, seed: int = 0, epoch: int = 0) -> list[dict]:
        """Every example the record supports for the enabled tasks."""
        rng = random.Random(f"{seed}:{epoch}:{record['id']}:expand")
        return derive(record, rng, list(self.weights), self.describer)

    def sample(self, record: dict, epoch: int = 0, index: int = 0, seed: int = 0, accept=None) -> dict:
        """One example: a task is picked by weight among those the record supports, then one of that
        task's examples at random. `accept(example) -> bool` filters candidates first (for instance
        "fits max_length"): rejected examples are simply not considered, and when none is left
        NoFittingExample is raised."""
        rng = random.Random(f"{seed}:{epoch}:{index}:{record['id']}")
        by_task: dict[str, list[dict]] = {}
        candidates = derive(record, rng, list(self.weights), self.describer)
        if not candidates:
            raise NoFittingExample(f"{record['id']} supports none of the enabled tasks {sorted(self.weights)}")
        for example in candidates:
            if accept is None or accept(example):
                by_task.setdefault(example["task"], []).append(example)
        if not by_task:
            raise NoFittingExample(f"{record['id']}: none of its {len(candidates)} candidate examples fits")
        task = rng.choices(list(by_task), weights=[self.weights[t] for t in by_task])[0]
        return rng.choice(by_task[task])
