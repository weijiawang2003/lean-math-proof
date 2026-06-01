#!/usr/bin/env python3
"""SX3 Part 7 — minimal-sufficient attribution over depth-2 sequence results.

Consumes one or more sx3_run_depth2_sequences result JSONs and classifies every
theorem (and every claimed sequence win) into an honest vocabulary, comparing
against RC2's deterministic credited mechanism (the per-theorem controls the
runner executes: simp / simp_all / aesop / classical<;>aesop / simp[Set.ite]).

Per-theorem attribution:
  TRUE_DEPTH2_SEQUENCE_WIN — a gated depth-2 sequence solved it AND every control
                             failed (beyond RC2 single-shot + bare baselines).
  RC2_ALREADY_SOLVED       — rc2_baseline_finished is True.
  SINGLE_STEP_DUPLICATE    — single-shot `simp [Set.ite]` control solved it
                             (belongs to RC2, not SX3).
  BASELINE_DUPLICATE       — a bare baseline (simp/simp_all/aesop/classical) solved it.
  PARSER_ARTIFACT          — sequences only failed via parse/recursion limits; no win.
  SOURCE_SPECIFIC          — the winning sequence copies a theorem-specific source
                             proof / named rw bridge (none of the SX3 families do this
                             by construction; flagged if detected).
  NO_WIN                   — nothing closed it (not a win, not parser-limited).
  NEEDS_REVIEW             — a win exists but trace is incomplete / inconsistent.

The SX3 families are all GENERIC tactic batteries (simp[Set.ite]<;>aesop, ext<;>...,
constructor<;>..., antisymm<;>...), so SOURCE_SPECIFIC is structurally impossible for
them; the check is still applied and reported for completeness.
"""
from __future__ import annotations
import argparse
import json

BASELINE_CONTROLS = {"simp", "simp_all", "aesop", "classical <;> aesop"}
SINGLE_STEP_CONTROL = "simp [Set.ite]"
# generic tactic tokens an SX3 family may legitimately contain
GENERIC_TOKENS = {"simp", "simp_all", "aesop", "ext", "intro", "constructor",
                  "Set.ite", "Set.Subset.antisymm", "classical", "x", "h"}


def winning_sequence(r):
    seq_wins = [w for w in r.get("wins", []) if w.get("kind") == "sequence"]
    return seq_wins[0] if seq_wins else None


def looks_source_specific(tac):
    """A generic SX3 sequence references only generic tactics/lemmas. Flag if it
    contains a theorem-specific rw bridge or a non-generic dotted lemma name."""
    if "rw " in tac or "rw[" in tac:
        return True
    # any dotted identifier that is not in the generic allow-list
    import re
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", tac):
        if "." in tok and tok not in {"Set.ite", "Set.Subset.antisymm",
                                       "Multiset.mem_toFinset"}:
            return True
    return False


def attribute(r):
    controls = r.get("controls", [])
    ctl_solved = {c["tactic"] for c in controls if c.get("solved")}
    seq_win = winning_sequence(r)

    if r.get("rc2_baseline_finished") is True:
        return "RC2_ALREADY_SOLVED", None
    if SINGLE_STEP_CONTROL in ctl_solved:
        return "SINGLE_STEP_DUPLICATE", seq_win["family"] if seq_win else None
    if ctl_solved & BASELINE_CONTROLS:
        return "BASELINE_DUPLICATE", seq_win["family"] if seq_win else None
    if seq_win:
        if looks_source_specific(seq_win["tactic"]):
            return "SOURCE_SPECIFIC", seq_win["family"]
        # genuine: a depth-2 sequence solved it and no control did
        return "TRUE_DEPTH2_SEQUENCE_WIN", seq_win["family"]
    # no win
    seq_tried = r.get("gated_sequences_tried", [])
    outs = {s.get("outcome") for s in seq_tried}
    if seq_tried and outs <= {"parse_error", "max_recursion", "timeout_inner",
                              "ext_not_applicable", "no_goals", "exception"}:
        return "PARSER_ARTIFACT", None
    if not r.get("live") or r.get("setup_error"):
        return "NEEDS_REVIEW", None
    return "NO_WIN", None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--registry",
                   default="project/evolve/experiments/sx3/sx3_sequence_registry.json")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    registry = json.load(open(args.registry))
    fam_keys = list(registry["families"].keys())

    per_theorem = []
    fam_stats = {k: {"true_wins": [], "single_step_dup": [], "baseline_dup": [],
                     "source_specific": [], "off_gate_emissions": 0,
                     "parser_artifact": [], "no_win": [], "needs_review": [],
                     "rc2_already": []} for k in fam_keys}

    for path in args.results:
        d = json.load(open(path))
        surface = d.get("cases_file", path)
        for r in d.get("results", []):
            label, fam = attribute(r)
            seq_win = winning_sequence(r)
            rec = {"full_name": r.get("full_name"), "surface": surface,
                   "role": r.get("role"),
                   "attribution": label,
                   "winning_family": fam,
                   "winning_sequence": seq_win["tactic"] if seq_win else None,
                   "controls_solved": [c["tactic"] for c in r.get("controls", []) if c.get("solved")],
                   "live": r.get("live")}
            per_theorem.append(rec)
            # bucket per family (use the family that produced the win when applicable)
            if fam and fam in fam_stats:
                bucket = {"TRUE_DEPTH2_SEQUENCE_WIN": "true_wins",
                          "SINGLE_STEP_DUPLICATE": "single_step_dup",
                          "BASELINE_DUPLICATE": "baseline_dup",
                          "SOURCE_SPECIFIC": "source_specific"}.get(label)
                if bucket:
                    fam_stats[fam][bucket].append(r.get("full_name"))

    # off-gate emissions: any emitted sequence on a non-Set / forbidden case.
    # detect via gate_decisions on negative/smoke surfaces.
    off_gate = {k: [] for k in fam_keys}
    for path in args.results:
        d = json.load(open(path))
        is_neg = "negative" in path or "smoke" in path
        for r in d.get("results", []):
            ns = r.get("namespace", "")
            for g in r.get("gate_decisions", []):
                if g.get("emitted") and (is_neg or ns not in ("Set",)):
                    off_gate[g["family"]].append(r.get("full_name"))
                    if g["family"] in fam_stats:
                        fam_stats[g["family"]]["off_gate_emissions"] += 1

    # roll up attribution histogram
    hist = {}
    for rec in per_theorem:
        hist[rec["attribution"]] = hist.get(rec["attribution"], 0) + 1

    true_wins = sorted({r["full_name"] for r in per_theorem
                        if r["attribution"] == "TRUE_DEPTH2_SEQUENCE_WIN"})
    # split true wins into deferred-known vs fresh
    deferred4 = {"Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl",
                 "Set.ite_inter_compl_self"}
    deferred_wins = sorted(w for w in true_wins if w in deferred4)
    fresh_wins = sorted(w for w in true_wins if w not in deferred4)

    out = {
        "inputs": args.results,
        "attribution_histogram": hist,
        "true_depth2_wins": true_wins,
        "deferred_known_true_wins": deferred_wins,
        "fresh_true_wins": fresh_wins,
        "num_off_gate_emissions": sum(len(v) for v in off_gate.values()),
        "off_gate_by_family": {k: v for k, v in off_gate.items() if v},
        "per_family": {k: {
            "true_wins": v["true_wins"],
            "num_true_wins": len(v["true_wins"]),
            "single_step_duplicates": v["single_step_dup"],
            "baseline_duplicates": v["baseline_dup"],
            "source_specific": v["source_specific"],
            "off_gate_emissions": v["off_gate_emissions"],
        } for k, v in fam_stats.items()},
        "per_theorem": sorted(per_theorem, key=lambda x: (x["surface"], x["full_name"])),
        "rc3_candidate_family": "SX3_SET_ITE_AESOP",
        "rc3_candidate_assessment": _assess(fam_stats, fresh_wins, deferred_wins, off_gate),
        "note": "TRUE_DEPTH2_SEQUENCE_WIN = depth-2 sequence solved & ALL controls "
                "(incl single-shot simp[Set.ite] = RC2 credited mechanism) failed. "
                "SX3 families are generic batteries -> SOURCE_SPECIFIC structurally 0.",
    }
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    _write_md(out, args.out_md)
    print(f"[sx3:attr] hist={hist}")
    print(f"[sx3:attr] deferred_known_true={deferred_wins}")
    print(f"[sx3:attr] fresh_true={fresh_wins}")
    print(f"[sx3:attr] off_gate={out['num_off_gate_emissions']}")
    print(f"[sx3:attr] SX3_SET_ITE_AESOP -> {out['rc3_candidate_assessment']['verdict']}")


def _assess(fam_stats, fresh_wins, deferred_wins, off_gate):
    f = "SX3_SET_ITE_AESOP"
    fs = fam_stats[f]
    fresh_for_fam = [w for w in fs["true_wins"]
                     if w not in {"Set.ite_inter", "Set.ite_inter_self",
                                  "Set.ite_compl", "Set.ite_inter_compl_self"}]
    deferred_for_fam = [w for w in fs["true_wins"] if w not in fresh_for_fam]
    n_offgate = len(off_gate.get(f, []))
    reproduced4 = len(deferred_for_fam) == 4
    has_fresh = len(fresh_for_fam) >= 1
    if n_offgate > 0:
        verdict = "REJECT_OFF_GATE"
    elif reproduced4 and has_fresh:
        verdict = "RC3_CANDIDATE_CONFIRMED"
    elif reproduced4 and not has_fresh:
        verdict = "TRAINING_DATA_ONLY"
    elif not reproduced4 and has_fresh:
        verdict = "KEEP_SX3_EXPERIMENTAL"
    else:
        verdict = "REJECT_NO_FRESH_DELTA"
    return {"family": f, "reproduced_deferred4": reproduced4,
            "deferred_true_wins": deferred_for_fam,
            "fresh_true_wins": fresh_for_fam, "off_gate": n_offgate,
            "verdict": verdict}


def _write_md(out, path):
    L = ["# SX3 Minimal-Sufficient Attribution", ""]
    L.append(f"- inputs: {', '.join('`'+p.split('/')[-1]+'`' for p in out['inputs'])}")
    L.append(f"- attribution histogram: `{out['attribution_histogram']}`")
    L.append(f"- **deferred-known true depth-2 wins ({len(out['deferred_known_true_wins'])}/4):** "
             f"{', '.join('`'+w+'`' for w in out['deferred_known_true_wins']) or '(none)'}")
    L.append(f"- **fresh true depth-2 wins ({len(out['fresh_true_wins'])}):** "
             f"{', '.join('`'+w+'`' for w in out['fresh_true_wins']) or '(none)'}")
    L.append(f"- off-gate emissions: **{out['num_off_gate_emissions']}**")
    a = out["rc3_candidate_assessment"]
    L.append(f"- SX3_SET_ITE_AESOP verdict: **{a['verdict']}** "
             f"(reproduced_deferred4={a['reproduced_deferred4']}, "
             f"fresh={len(a['fresh_true_wins'])}, off_gate={a['off_gate']})")
    L.append("")
    L.append("## Per-theorem attribution")
    L.append("| theorem | role | attribution | winning sequence | controls solved |")
    L.append("|---|---|---|---|---|")
    for r in out["per_theorem"]:
        L.append(f"| `{r['full_name']}` | {r.get('role','')} | **{r['attribution']}** | "
                 f"`{(r.get('winning_sequence') or '')[:34]}` | "
                 f"{', '.join(r['controls_solved']) or '-'} |")
    L.append("")
    L.append("## Per-family")
    L.append("| family | true wins | single-step dup | baseline dup | source-spec | off-gate |")
    L.append("|---|---|---|---|---|---|")
    for k, v in out["per_family"].items():
        L.append(f"| {k} | {v['num_true_wins']} | {len(v['single_step_duplicates'])} | "
                 f"{len(v['baseline_duplicates'])} | {len(v['source_specific'])} | "
                 f"{v['off_gate_emissions']} |")
    L.append("")
    L.append("> " + out["note"])
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
