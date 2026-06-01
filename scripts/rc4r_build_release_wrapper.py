#!/usr/bin/env python3
"""RC4R Part 2 — build the clean RC4 release-candidate wrapper.

Constructs the RC4 wrapper as a clean copy of the frozen RC2 wrapper with ONLY the validated
RC4D deployable actions added (the 15 component tactics prepended to priority_templates["any"]
+ their name-prefix gates), plus a release-candidate metadata block. RC2's fields are preserved
exactly. The added tactics + gates are taken verbatim from the validated RC4D candidate wrapper
(which carries the gate-prefix fix: RC2's gate matches full_name.startswith(prefix)). The diff
against RC2 is recorded and asserted to contain ONLY the intended additions — any unrelated RC2
field change is a hard reject.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# component label per added tactic (for the diff report's component mapping)
RC4A_TACTICS = {"simp [Finset.disjUnion]", "simp [Monotone, MonotoneOn]",
                "simp [Antitone, AntitoneOn]", "simp [StrictMono, StrictMonoOn]",
                "simp [StrictAnti, StrictAntiOn]"}
RC4B_TACTICS = {"simp [Set.disjoint_left]", "simp [Set.disjoint_left] <;> aesop",
                "simp [Multiset.disjoint_left]", "simp [Multiset.disjoint_left] <;> aesop"}
RC4C_TACTICS = {"simp [Multiset.disjoint_right]", "simp [Multiset.disjoint_right] <;> aesop",
                "simp [Set.subset_pair_iff_eq]", "simp [Set.subset_pair_iff_eq] <;> aesop",
                "simp [List.forall_iff_forall_mem]", "simp [List.forall_iff_forall_mem] <;> aesop"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _component_of(tac):
    if tac in RC4A_TACTICS:
        return "RC4A"
    if tac in RC4B_TACTICS:
        return "RC4B"
    if tac in RC4C_TACTICS:
        return "RC4C_residue"
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-wrapper", required=True)
    ap.add_argument("--rc4d-wrapper", required=True)
    ap.add_argument("--rc4d-policy", required=True)
    ap.add_argument("--out-wrapper", required=True)
    ap.add_argument("--out-diff-json", required=True)
    ap.add_argument("--out-diff-md", required=True)
    args = ap.parse_args()

    rc2 = json.load(open(_p(args.rc2_wrapper)))
    rc4d = json.load(open(_p(args.rc4d_wrapper)))
    policy = json.load(open(_p(args.rc4d_policy)))

    # ---- build clean RC4 wrapper from RC2 ----
    w = copy.deepcopy(rc2)
    rc2_any = list(rc2.get("priority_templates", {}).get("any", []))
    rc4d_any = list(rc4d.get("priority_templates", {}).get("any", []))
    added_tactics = [t for t in rc4d_any if t not in rc2_any]
    # prepend the RC4 tactics (preserve RC4D ordering: RC4A, RC4B, RC4C_residue)
    new_any = added_tactics + [t for t in rc2_any if t not in added_tactics]
    w.setdefault("priority_templates", {})["any"] = new_any

    rc2_gates = rc2.get("theorem_name_tactic_gates", {})
    rc4d_gates = rc4d.get("theorem_name_tactic_gates", {})
    added_gates = {k: v for k, v in rc4d_gates.items() if k not in rc2_gates}
    g = dict(rc2_gates)
    g.update(added_gates)
    w["theorem_name_tactic_gates"] = g

    # drop the rc4d metadata key if it leaked via deepcopy of rc2 (it won't, rc2 has none)
    w.pop("_rc4d_candidate_metadata", None)
    w["_rc4_release_candidate_metadata"] = {
        "family": "rc4_release_candidate",
        "base": "RC2",
        "base_wrapper": args.rc2_wrapper,
        "composition": "RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue (= validated RC4D)",
        "decision_source": "RC4D_COMPOSITION_CANDIDATE_CONFIRMED",
        "promotion_allowed": False,
        "status": "release_candidate_off_by_default_pending_owner_approval",
        "added_tactics_priority_any_prepended": added_tactics,
        "theorem_name_tactic_gates_added": added_gates,
        "component_mapping": {t: _component_of(t) for t in added_tactics},
        "rc1_wrapper_untouched": True, "rc2_wrapper_untouched": True,
        "ns24_router_untouched": True,
        "gate_semantics_note": "theorem_name_tactic_gates match full_name.startswith(prefix); "
                               "non-gate-firing theorems are byte-identical to RC2 (purely additive).",
    }
    os.makedirs(os.path.dirname(_p(args.out_wrapper)), exist_ok=True)
    json.dump(w, open(_p(args.out_wrapper), "w"), ensure_ascii=False, indent=2)

    # ---- exact diff against RC2 ----
    rc2_keys, w_keys = set(rc2), set(w)
    added_keys = sorted(w_keys - rc2_keys)
    removed_keys = sorted(rc2_keys - w_keys)
    modified_fields, unchanged_fields = [], []
    for k in sorted(rc2_keys & w_keys):
        if rc2[k] != w[k]:
            modified_fields.append(k)
        else:
            unchanged_fields.append(k)

    # validate: only the two intended fields changed, only the one metadata key added
    intended_modified = {"priority_templates", "theorem_name_tactic_gates"}
    unrelated_changes = [k for k in modified_fields if k not in intended_modified]
    unrelated_added = [k for k in added_keys if k != "_rc4_release_candidate_metadata"]
    # deep-check priority_templates: only "any" changed, only by the added tactics
    pt_clean = True
    for sub in set(rc2.get("priority_templates", {})) | set(w.get("priority_templates", {})):
        if sub == "any":
            continue
        if rc2.get("priority_templates", {}).get(sub) != w.get("priority_templates", {}).get(sub):
            pt_clean = False
    any_only_added = set(new_any) - set(rc2_any) == set(added_tactics) and \
        all(t in new_any for t in rc2_any)
    gates_clean = all(rc2_gates[k] == g[k] for k in rc2_gates)  # RC2 gates preserved
    clean = (not unrelated_changes and not unrelated_added and not removed_keys
             and pt_clean and any_only_added and gates_clean)

    by_comp = {}
    for t in added_tactics:
        by_comp.setdefault(_component_of(t), []).append(t)

    diff = {
        "generated_by": "scripts/rc4r_build_release_wrapper.py",
        "rc2_wrapper": args.rc2_wrapper, "rc4_wrapper": args.out_wrapper,
        "added_tactics": added_tactics, "num_added_tactics": len(added_tactics),
        "added_gates": added_gates, "num_added_gates": len(added_gates),
        "added_top_level_keys": added_keys, "removed_top_level_keys": removed_keys,
        "modified_fields": modified_fields, "unchanged_rc2_fields": unchanged_fields,
        "component_mapping": by_comp,
        "rc2_priority_any_preserved": all(t in new_any for t in rc2_any),
        "rc2_gates_preserved": gates_clean,
        "unrelated_changes": unrelated_changes + unrelated_added,
        "diff_clean": clean,
        "verdict": "WRAPPER_DIFF_CLEAN" if clean else "REJECT_UNRELATED_CHANGES",
    }
    json.dump(diff, open(_p(args.out_diff_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4 release wrapper diff vs RC2", "",
          f"- verdict: **{diff['verdict']}**",
          f"- added tactics ({len(added_tactics)}): prepended to `priority_templates['any']`",
          f"- added gates: {len(added_gates)} | added top-level keys: {added_keys}",
          f"- modified fields (intended = priority_templates, theorem_name_tactic_gates): {modified_fields}",
          f"- removed keys: {removed_keys} | unrelated changes: {diff['unrelated_changes']}",
          f"- RC2 priority.any preserved: {diff['rc2_priority_any_preserved']} | "
          f"RC2 gates preserved: {gates_clean}", "",
          "## Component mapping", "", "| component | tactics |", "|---|---|"]
    for c in ("RC4A", "RC4B", "RC4C_residue"):
        md.append(f"| {c} | {by_comp.get(c, [])} |")
    md += ["", f"- unchanged RC2 fields ({len(unchanged_fields)}): {unchanged_fields}"]
    open(_p(args.out_diff_md), "w").write("\n".join(md) + "\n")

    print(f"[rc4r-wrapper] {diff['verdict']} | added {len(added_tactics)} tactics / "
          f"{len(added_gates)} gates | unrelated_changes={diff['unrelated_changes']}")
    print(f"[rc4r-wrapper] component_mapping: { {c: len(v) for c, v in by_comp.items()} }")
    if not clean:
        raise SystemExit("REJECT: wrapper diff includes unrelated changes")


if __name__ == "__main__":
    main()
