#!/usr/bin/env python3
"""FLI3 Part 8b — determinism: the gate is a pure function; re-run twice and hash."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fli3_gate as G
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(*a): return os.path.join(_REPO, *a)
def _hash(items):
    blob = [(it["theorem"], G.gate(it)["gate"], tuple(G.gate(it)["action_templates"])) for it in items]
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", required=True)
    ap.add_argument("--out-json", required=True); a = ap.parse_args()
    items = json.load(open(_p(a.sets)))["items"]
    h1, h2 = _hash(items), _hash(items)
    out = {"generated_by": "scripts/fli3_determinism_check.py", "gate_hash_run1": h1,
           "gate_hash_run2": h2, "deterministic": h1 == h2,
           "note": "Gate is a pure function of (theorem, namespace, statement, family); live "
                   "candidate wins were re-run for robustness in Part 6."}
    json.dump(out, open(_p(a.out_json), "w"), indent=2)
    print(f"[fli3-determinism] deterministic={h1==h2} hash={h1}")
if __name__ == "__main__": main()
