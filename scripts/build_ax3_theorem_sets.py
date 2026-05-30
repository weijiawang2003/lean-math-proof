"""AX3 Stage 2 — write the AX3 Multiset theorem sets.

Reads project/data/ax3_multiset_heldout_audit_meta.json and writes the four
disjoint sets to project/evolve/routing/ax3_theorem_sets.json (loaded by
tasks._load_ax3_sets). All candidates are confirmed-available, fresh, and
held out from WX3 / prior arcs.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "project/data/ax3_multiset_heldout_audit_meta.json"
OUT = ROOT / "project/evolve/routing/ax3_theorem_sets.json"


def main() -> None:
    m = json.loads(AUDIT.read_text(encoding="utf-8"))
    sets = {name: [{"file_path": t["file_path"], "full_name": t["full_name"]}
                   for t in items]
            for name, items in m["splits"].items()}
    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
