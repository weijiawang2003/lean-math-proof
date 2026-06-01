#!/usr/bin/env python3
"""Shared utilities for the SF1 Scalable Frontier Miner pipeline.

This module is imported by the ``sf1_*.py`` stage scripts. It is deterministic
and side-effect free except for the explicit write helpers. It NEVER reads or
writes any production config (RC1 wrapper, NS9 genome, NS24 router, REL1
reports).

Public API
----------
- read_json_or_jsonl(path)          -> list[Any]   (robust JSON / JSONL loader)
- write_jsonl(rows, path)           -> int         (atomic-ish JSONL writer)
- ensure_parent_dir(path)           -> str
- stable_hash(value)                -> str         (deterministic, salt-free)
- deterministic_sample(items, k, *, seed, key=None) -> list
- extract_decl_names_from_record(rec, key_hints=None) -> list[str]
- load_consumed_decl_names(sources, *, key_hints=None, recursive=True,
                           python_string_scan=False, verbose=False) -> (set, dict)

A "decl name" is a Lean declaration name such as ``Multiset.map_cons`` or
``Nat.add_mod_eq_ite``: one or more dotted identifier segments. The
``DECL_NAME_RE`` heuristic is used to keep regex-fallback extraction from
sweeping in tactic strings or prose.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re

# A Lean identifier segment: letter/underscore start, then word chars / prime /
# subscript-ish unicode digits. Decl names are usually dotted (>=1 dot), but we
# also accept bare single identifiers when they come from a trusted key.
_IDENT = r"[A-Za-z_][A-Za-z0-9_'₀-₉°-ÿ]*"
DECL_NAME_RE = re.compile(rf"^{_IDENT}(?:\.{_IDENT})+$")
BARE_IDENT_RE = re.compile(rf"^{_IDENT}$")

# Keys, in priority order, that commonly hold a theorem/decl name across the
# repo's experiment artifacts, eval surfaces and benchmark logs.
DEFAULT_KEY_HINTS = (
    "decl_name",
    "theorem_name",
    "theorem_full_name",
    "full_name",
    "thm_name",
    "thm",
    "theorem",
    "name",
)

# Keys whose value is a NESTED record that itself carries the name.
_NESTED_KEYS = ("theorem", "thm", "problem", "goal", "task", "record")


# --------------------------------------------------------------------------- #
# filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_parent_dir(path):
    """Create the parent directory of ``path`` if needed; return ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def read_json_or_jsonl(path):
    """Load a .json / .jsonl / .txt file into a list of records.

    - ``.json``  : a top-level list is returned as-is; a dict is wrapped in a
                   one-element list; a scalar is wrapped too. Falls back to
                   line-by-line JSONL parsing if a single ``json.load`` fails.
    - ``.jsonl`` : one JSON value per non-blank line.
    - ``.txt``   : one record (string) per non-blank, non-``#`` line.
    - other      : best-effort — try JSON, then JSONL, then TXT lines.
    """
    if not os.path.isfile(path):
        return []
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    def _as_lines_jsonl(blob):
        out = []
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a stray non-JSON line by keeping it as a raw string
                out.append(line)
        return out

    def _as_txt(blob):
        out = []
        for line in blob.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        return out

    if ext == ".txt":
        return _as_txt(text)
    if ext == ".jsonl":
        return _as_lines_jsonl(text)
    # .json or unknown: try whole-file JSON first.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        rows = _as_lines_jsonl(text)
        # If JSONL parsing produced mostly raw strings, treat as TXT instead.
        if rows and all(isinstance(r, str) for r in rows):
            return _as_txt(text)
        return rows
    if isinstance(obj, list):
        return obj
    return [obj]


def write_jsonl(rows, path):
    """Write ``rows`` as JSONL to ``path``; return the number of rows written."""
    ensure_parent_dir(path)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# determinism helpers
# --------------------------------------------------------------------------- #
def stable_hash(value):
    """Return a deterministic hex digest for ``value`` (salt-free, cross-run).

    Uses a canonical JSON encoding so dict key order does not affect the hash.
    Python's built-in ``hash()`` is salted per-process and must NOT be used for
    reproducible sampling.
    """
    if isinstance(value, (dict, list, tuple)):
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    else:
        blob = str(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def deterministic_sample(items, k, *, seed, key=None):
    """Return up to ``k`` items chosen deterministically from ``items``.

    The selection depends only on (item-key, seed), never on input order or
    process state, so the same inputs always yield the same sample. Items are
    ranked by ``stable_hash((seed, key(item)))`` and the lowest ``k`` are kept,
    then returned in that stable ranked order.
    """
    items = list(items)
    if k is None or k < 0 or k >= len(items):
        ranked = sorted(items, key=lambda it: stable_hash([seed, _keyof(it, key)]))
        return ranked
    ranked = sorted(items, key=lambda it: stable_hash([seed, _keyof(it, key)]))
    return ranked[:k]


def _keyof(item, key):
    if key is None:
        return item if isinstance(item, (str, int, float, bool)) else stable_hash(item)
    return key(item)


# --------------------------------------------------------------------------- #
# decl-name extraction
# --------------------------------------------------------------------------- #
def _looks_like_decl_name(s, *, allow_bare=False):
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > 200:
        return False
    if DECL_NAME_RE.match(s):
        return True
    if allow_bare and BARE_IDENT_RE.match(s):
        return True
    return False


def extract_decl_names_from_record(rec, key_hints=None, *, scan_keys=True,
                                   max_depth=6):
    """Extract decl name(s) from a single record (str / dict / list).

    - str  : returned if it looks like a dotted decl name.
    - dict : values under any ``key_hints`` key are taken as names (bare single
             identifiers allowed under these trusted keys); dotted-decl-name
             *keys* are taken too when ``scan_keys`` is set (this is how
             per-theorem result files keyed by theorem name are harvested); the
             record is then recursed into up to ``max_depth``.
    - list : flattened recursively.

    Recursion is depth-bounded so a deeply nested config blob cannot blow up.
    Only dotted names are accepted via the regex fallback, which keeps tactic
    strings and prose out of the set.
    """
    hints = tuple(key_hints) if key_hints else DEFAULT_KEY_HINTS
    out = []

    def _add_value(v, *, trusted):
        if isinstance(v, str):
            if _looks_like_decl_name(v, allow_bare=trusted):
                out.append(v.strip())
        elif isinstance(v, list):
            for x in v:
                _add_value(x, trusted=trusted)

    def _walk(node, depth):
        if depth < 0:
            return
        if isinstance(node, str):
            _add_value(node, trusted=False)
        elif isinstance(node, list):
            for x in node:
                _walk(x, depth - 1)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in hints:
                    _add_value(v, trusted=True)
                if scan_keys and isinstance(k, str) and _looks_like_decl_name(k):
                    out.append(k.strip())
                _walk(v, depth - 1)

    _walk(rec, max_depth)
    return out


# --------------------------------------------------------------------------- #
# python-source string scan (for benchmark_specs.py / tasks.py style files)
# --------------------------------------------------------------------------- #
_PY_STRING_RE = re.compile(r"""(['"])((?:\\.|(?!\1).)*?)\1""")


def _scan_python_string_literals(path):
    """Yield decl-name-looking string literals from a .py source file."""
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    for _, body in _PY_STRING_RE.findall(text):
        if _looks_like_decl_name(body):
            out.append(body)
    return out


# --------------------------------------------------------------------------- #
# consumed-surface loader
# --------------------------------------------------------------------------- #
_DATA_EXTS = (".json", ".jsonl", ".txt")


def _expand_source(src, *, recursive):
    """Resolve a source entry (file / dir / glob) to a list of concrete paths."""
    paths = []
    if os.path.isdir(src):
        pattern = os.path.join(src, "**", "*") if recursive else os.path.join(src, "*")
        for p in glob.glob(pattern, recursive=recursive):
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in _DATA_EXTS:
                paths.append(p)
    elif any(ch in src for ch in "*?[]"):
        for p in glob.glob(src, recursive=recursive):
            if os.path.isfile(p):
                paths.append(p)
    elif os.path.isfile(src):
        paths.append(src)
    return sorted(set(paths))


def load_consumed_decl_names(sources, *, key_hints=None, recursive=True,
                             python_string_scan=False, verbose=False):
    """Build a consumed-theorem set from a list of source entries.

    ``sources`` may be:
      - a directory   -> recursively scanned for *.json / *.jsonl / *.txt
      - a glob string -> expanded
      - a file path   -> parsed directly (json/jsonl/txt; .py if scan enabled)

    Returns ``(names: set[str], ledger: dict)`` where ``ledger`` maps each
    resolved source path to the count of decl names it contributed (and lists
    sources that were missing / empty). Missing paths are skipped, never raised.
    """
    names = set()
    ledger = {"per_path": {}, "missing": [], "scanned_paths": 0,
              "python_scanned": []}

    for src in sources:
        # Optional python-source scanning is opt-in and handled per explicit file.
        if python_string_scan and src.endswith(".py"):
            if os.path.isfile(src):
                found = _scan_python_string_literals(src)
                ledger["python_scanned"].append(src)
                ledger["per_path"][src] = len(found)
                names.update(found)
                ledger["scanned_paths"] += 1
            else:
                ledger["missing"].append(src)
            continue

        resolved = _expand_source(src, recursive=recursive)
        if not resolved:
            ledger["missing"].append(src)
            if verbose:
                print(f"[sf1_common] WARN: no files for source: {src}")
            continue
        for path in resolved:
            rows = read_json_or_jsonl(path)
            found = []
            for rec in rows:
                found.extend(extract_decl_names_from_record(rec, key_hints))
            ledger["per_path"][path] = len(found)
            ledger["scanned_paths"] += 1
            names.update(found)

    ledger["total_names"] = len(names)
    return names, ledger
