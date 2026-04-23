from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core_types import TransitionRecord

# Required keys when writing a raw dict (rather than a TransitionRecord).
REQUIRED_TRACE_KEYS = frozenset({
    "file_path",
    "full_name",
    "state_pp",
    "tactic",
    "result_kind",
    "proof_finished",
})


def append_jsonl(path: str, record: TransitionRecord | dict) -> None:
    if isinstance(record, TransitionRecord):
        obj = record.to_dict()
    else:
        missing = REQUIRED_TRACE_KEYS - record.keys()
        if missing:
            raise ValueError(
                f"Trace dict missing required key(s): {sorted(missing)}. "
                f"Use TransitionRecord or include all of: {sorted(REQUIRED_TRACE_KEYS)}"
            )
        obj = record
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
