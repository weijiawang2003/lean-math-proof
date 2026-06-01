#!/usr/bin/env python3
"""RC4D Part 2 — build the component manifest.

Pure analysis (no live Lean) over the three validated components' policies +
minimal_attribution + candidate_results. Lists which actions each component contributes to
RC4D, which RC4C actions are excluded (overlap / simp-only duplicate), the per-residue
deploy-form decision, known wins per component, and the EXPECTED theorem-level overlap
between RC4B and RC4C_residue (the de-dup that ordered attribution will enforce).

RC4C residue decision per candidate lemma (Multiset.disjoint_right / Set.subset_pair_iff_eq /
List.forall_iff_forall_mem):
  INCLUDE_AS_DEPTH1_SIMP        genuine win and `simp [L]` alone closes it.
  INCLUDE_AS_DEPTH2_SIMP_AESOP  genuine depth-2 (simp[L] alone fails, `<;> aesop` closes).
  EXCLUDE_SCHEMA_UNSTABLE       no deployable schema representation.
  EXCLUDE_OVERLAP               lemma is an RC4B action.
  EXCLUDE_DUPLICATE             SIMP_ONLY_DUPLICATE rolled into depth-1 elsewhere / dropped.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RC4A_DIR = "project/evolve/experiments/rc4_candidates/def_unfold_simp"
RC4B_DIR = "project/evolve/experiments/rc4_candidates/disjoint_left_bridge"
RC4C_DIR = "project/evolve/experiments/rc4_candidates/d2_simp_aesop"

# residue candidate lemmas (the task's three) -> the RC4C action that carries each
RESIDUE_LEMMAS = {
    "Multiset.disjoint_right": "MULTISET_DISJOINT_RIGHT_D2",
    "Set.subset_pair_iff_eq": "SET_SUBSET_PAIR_D2",
    "List.forall_iff_forall_mem": "LIST_FORALL_D2",
}
RC4B_LEMMAS = ("Set.disjoint_left", "Multiset.disjoint_left")


def _p(*a):
    return os.path.join(_REPO, *a)


def _j(*a):
    return json.load(open(_p(*a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    a_pol = _j(RC4A_DIR, "def_unfold_simp_policy.json")
    a_attr = _j(RC4A_DIR, "out", "minimal_attribution.json")
    b_pol = _j(RC4B_DIR, "disjoint_left_bridge_policy.json")
    b_attr = _j(RC4B_DIR, "out", "minimal_attribution.json")
    c_pol = _j(RC4C_DIR, "d2_simp_aesop_policy.json")
    c_attr = _j(RC4C_DIR, "out", "minimal_attribution.json")
    c_smoke = _j(RC4C_DIR, "out", "schema_wrapper_smoke.json")

    # ---- RC4A ----
    rc4a = {
        "decision": "RC4A_CANDIDATE_CONFIRMED",
        "included_actions": [{"name": "def_unfold_simp_allowlist",
                              "emit": "simp [<allowlisted defs in goal>]",
                              "allowlist": a_pol["validated_def_allowlist"]}],
        "known_wins": a_attr["true_def_unfold_win_targets"],
        "num_known_wins": a_attr["num_true_def_unfold_wins"],
    }

    # ---- RC4B ----
    rc4b = {
        "decision": "RC4B_CANDIDATE_CONFIRMED",
        "included_actions": [{"name": a["name"], "tactic": a["tactic"], "lemma": a["lemma"]}
                             for a in b_pol["actions"]],
        "bridge_lemmas": b_pol["bridge_lemmas"],
        "known_wins": sorted(set(b_attr["split"]["Set_true_wins"] + b_attr["split"]["Multiset_true_wins"])),
        "num_known_wins": b_attr["num_true_bridge_wins"],
        "fresh_holdout_wins": b_attr["split"]["fresh_holdout_true_wins"],
    }
    rc4b_win_set = set(rc4b["known_wins"])

    # ---- RC4C residue decisions ----
    # winning lemma per pure-RC4C theorem (from records)
    rec_by_thm = {r["full_name"]: r for r in c_attr["records"]}
    pure = set(c_attr["split"]["pure_rc4c_true_wins"])
    simp_only = set(c_attr["split"]["simp_only_duplicates"])

    def lemma_decision(lemma, action_name):
        # find pure-RC4C theorems whose genuine non-overlap d2 used this lemma
        wins, fresh = [], []
        depth1_alone = False
        for fn in pure:
            r = rec_by_thm.get(fn, {})
            if lemma in (r.get("nonoverlap_genuine_d2_lemmas") or []):
                wins.append(fn)
                if r.get("fresh"):
                    fresh.append(fn)
            # is simp[L] alone enough anywhere? -> would be depth-1
            pl = (r.get("per_lemma") or {}).get(lemma)
            if pl and pl.get("simp_only"):
                depth1_alone = True
        if not wins:
            return "EXCLUDE_SCHEMA_UNSTABLE", wins, fresh
        # genuine depth-2 across the board -> deploy RC4B-style (bare simp + combinator)
        decision = "INCLUDE_AS_DEPTH1_SIMP" if depth1_alone else "INCLUDE_AS_DEPTH2_SIMP_AESOP"
        return decision, wins, fresh

    residue = []
    for lemma, action in RESIDUE_LEMMAS.items():
        dec, wins, fresh = lemma_decision(lemma, action)
        residue.append({"lemma": lemma, "action": action, "decision": dec,
                        "theorem_wins": sorted(wins), "fresh_wins": sorted(fresh),
                        "deploy_tactics": [f"simp [{lemma}]", f"simp [{lemma}] <;> aesop"]})

    included_residue = [r for r in residue if r["decision"].startswith("INCLUDE")]

    # ---- excluded RC4C actions ----
    excluded_overlap = [{"action": a["name"], "lemma": a["lemma"], "reason": "EXCLUDE_OVERLAP",
                         "overlap_family": "RC4B"}
                        for a in c_pol["actions"] if a["lemma"] in RC4B_LEMMAS]
    excluded_simp_only = [{"action": "FINSET_BIUNION_SUBSET_D2", "lemma": "Finset.biUnion_subset",
                           "reason": "EXCLUDE_DUPLICATE",
                           "detail": "SIMP_ONLY_DUPLICATE: simp [Finset.biUnion_subset] closes "
                                     "Finset.biUnion_subset_iff_forall_subset alone (depth-1)."}]
    excluded_schema = [{"action": "RC4C fused `<;> aesop`-only deployment", "lemma": "(all residue)",
                        "reason": "schema_nonreproducible_in_original_form",
                        "detail": f"RC4C schema smoke solved {c_smoke['known_wins_solved_by_wrapper']}"
                                  f"/{c_smoke['known_wins_total']} known via the fused combinator. "
                                  "RC4D re-deploys residue RC4B-style (bare simp + combinator)."}]

    # ---- expected theorem-level RC4B ∩ RC4C overlap ----
    residue_thms = set()
    for r in included_residue:
        residue_thms |= set(r["theorem_wins"])
    overlap_thms = sorted(residue_thms & rc4b_win_set)
    residue_additive_thms = sorted(residue_thms - rc4b_win_set)

    rc4c_residue = {
        "decision": "RC4C_CONFIRMED_WITH_RC4B_OVERLAP",
        "residue_lemma_decisions": residue,
        "included_actions": [{"name": r["action"], "lemma": r["lemma"], "decision": r["decision"],
                              "tactics": r["deploy_tactics"]} for r in included_residue],
        "excluded_overlap_rc4b": excluded_overlap,
        "excluded_simp_only_duplicate": excluded_simp_only,
        "excluded_schema_nonreproducible": excluded_schema,
        "residue_theorem_wins": sorted(residue_thms),
        "theorem_overlap_with_rc4b": overlap_thms,
        "theorem_additive_over_rc4b": residue_additive_thms,
        "note": "RC4C_residue Multiset.disjoint_right wins land on theorems RC4B already solves "
                "via disjoint_left -> credited to RC4B under ordering. Genuinely-additive "
                "theorem coverage = " + str(residue_additive_thms),
    }

    manifest = {
        "generated_by": "scripts/rc4d_build_component_manifest.py",
        "family": "rc4d_composition",
        "base": "RC2",
        "ordering": ["RC4A", "RC4B", "RC4C_residue"],
        "components": {"RC4A": rc4a, "RC4B": rc4b, "RC4C_residue": rc4c_residue},
        "expected_overlap": {
            "rc4b_known_wins": len(rc4b_win_set),
            "rc4c_residue_theorem_wins": len(residue_thms),
            "rc4b_rc4c_theorem_overlap": overlap_thms,
            "rc4c_residue_additive_over_rc4b": residue_additive_thms,
            "expected_credited_components": {
                "RC4A": rc4a["num_known_wins"],
                "RC4B": rc4b["num_known_wins"],
                "RC4C_residue_additive": len(residue_additive_thms),
            },
        },
        "total_distinct_actions": (len(rc4a["included_actions"]) + len(rc4b["included_actions"])
                                   + len(included_residue)),
    }
    os.makedirs(os.path.dirname(_p(args.out_json)), exist_ok=True)
    json.dump(manifest, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4D component manifest", "",
          f"- ordering: {manifest['ordering']}",
          f"- total distinct actions: {manifest['total_distinct_actions']}", "",
          "## RC4A — def_unfold_simp (CONFIRMED)",
          f"- allowlist: {a_pol['validated_def_allowlist']}",
          f"- known wins ({rc4a['num_known_wins']}): {rc4a['known_wins']}", "",
          "## RC4B — disjoint_left bridge (CONFIRMED)",
          f"- actions: {[a['name'] for a in rc4b['included_actions']]}",
          f"- known wins ({rc4b['num_known_wins']}): {rc4b['known_wins']}",
          f"- fresh-holdout wins: {rc4b['fresh_holdout_wins']}", "",
          "## RC4C_residue (de-duplicated)",
          "### Residue lemma decisions", "",
          "| lemma | action | decision | wins | fresh |", "|---|---|---|---|---|"]
    for r in residue:
        md.append(f"| `{r['lemma']}` | {r['action']} | **{r['decision']}** | "
                  f"{len(r['theorem_wins'])} | {len(r['fresh_wins'])} |")
    md += ["", "### Excluded RC4C actions", "",
           "| action | lemma | reason |", "|---|---|---|"]
    for e in excluded_overlap + excluded_simp_only:
        md.append(f"| {e['action']} | `{e['lemma']}` | {e['reason']} |")
    md += ["", "### Theorem-level overlap (RC4B ∩ RC4C_residue)",
           f"- residue theorem wins: {sorted(residue_thms)}",
           f"- overlap with RC4B (credited to RC4B): {overlap_thms}",
           f"- **additive over RC4B (RC4C_residue credit): {residue_additive_thms}**", "",
           "## Expected credited components",
           f"- RC4A: {rc4a['num_known_wins']}",
           f"- RC4B: {rc4b['num_known_wins']}",
           f"- RC4C_residue additive: {len(residue_additive_thms)}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[rc4d-manifest] RC4A={rc4a['num_known_wins']} RC4B={rc4b['num_known_wins']} "
          f"RC4C_residue_actions={len(included_residue)} "
          f"residue_additive_over_rc4b={residue_additive_thms}")
    print(f"[rc4d-manifest] residue decisions: "
          f"{ {r['lemma']: r['decision'] for r in residue} }")
    print(f"[rc4d-manifest] overlap RC4B∩RC4C_residue: {overlap_thms}")


if __name__ == "__main__":
    main()
