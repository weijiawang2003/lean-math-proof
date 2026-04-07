from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core_types import TransitionRecord


def append_jsonl(path: str, record: TransitionRecord | dict) -> None:
    obj = record.to_dict() if isinstance(record, TransitionRecord) else record
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
