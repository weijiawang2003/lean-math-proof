#!/usr/bin/env python3
"""FLI3 Part 8c — schema smoke: build a HYPOTHETICAL FLI3 wrapper fragment, validate, DO NOT install."""
from __future__ import annotations
import argparse, json, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(*a): return os.path.join(_REPO, *a)
RC2 = "project/evolve/experiments/rc2_release/rc2_production_wrapper.json"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", required=True); a = ap.parse_args()
    # hypothetical fragment in the RC2 wrapper schema (priority_templates["any"] + gates)
    fragment = {
        "_fli3_candidate_fragment": True,
        "priority_templates_any_additions": [
            "simp [Finset.card_le_one] <;> aesop", "simp [Finset.filterMap]", "simp [Finset.map]",
            "simp [Finset.preimage]", "simp [Finset.subtype]", "simp [List.bidirectionalRec]"],
        "theorem_name_tactic_gates_additions": {
            "simp [Finset.card_le_one] <;> aesop": ["Finset."],
            "simp [Finset.filterMap]": ["Finset."], "simp [Finset.map]": ["Finset."],
            "simp [Finset.preimage]": ["Finset."], "simp [Finset.subtype]": ["Finset."],
            "simp [List.bidirectionalRec]": ["List."]},
    }
    checks = {"json_serializable": True, "rc2_wrapper_readable": os.path.exists(_p(RC2)),
              "fragment_keys_valid": all(k in fragment for k in
                  ("priority_templates_any_additions", "theorem_name_tactic_gates_additions")),
              "gates_reference_known_actions": set(fragment["theorem_name_tactic_gates_additions"])
                  .issubset(set(fragment["priority_templates_any_additions"])),
              "installed": False}
    # confirm it merges onto a COPY of the RC2 wrapper in memory without breaking schema
    try:
        rc2 = json.load(open(_p(RC2)))
        copy = json.loads(json.dumps(rc2))
        copy.setdefault("priority_templates", {}).setdefault("any", [])
        merged_any = list(copy["priority_templates"]["any"]) + fragment["priority_templates_any_additions"]
        copy["priority_templates"]["any"] = merged_any
        copy.setdefault("theorem_name_tactic_gates", {}).update(
            fragment["theorem_name_tactic_gates_additions"])
        json.dumps(copy)  # serializable?
        checks["merges_onto_rc2_copy"] = True
    except Exception as e:  # noqa: BLE001
        checks["merges_onto_rc2_copy"] = False
        checks["merge_error"] = str(e)[:160]
    ok = all(v for k, v in checks.items() if isinstance(v, bool) and k != "installed")
    out = {"generated_by": "scripts/fli3_schema_smoke.py", "schema_compatible": ok,
           "checks": checks, "fragment": fragment,
           "note": "Fragment validated in memory only; NOT written to any wrapper. No install."}
    json.dump(out, open(_p(a.out_json), "w"), indent=2)
    print(f"[fli3-schema] schema_compatible={ok} installed=False checks={checks}")
if __name__ == "__main__": main()
