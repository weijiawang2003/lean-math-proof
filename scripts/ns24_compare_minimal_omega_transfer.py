"""NS24 — minimal-omega transfer vs absorption analysis.

Compares each NS24 candidate (trained on NS23-repaired minimal-tactic
labels) against the NS22 omega_5x baseline (and NS12 where cached) on
the Int surface. Unlike NS22's analysis, NS24 partitions the 22-theorem
omega aggregate by *provenance under the repaired labels*:

  - old_ns22_omega   : 12 originally-fallback_omega, omega-minimal.
                       These were already in NS22's omega training set.
  - relabeled_iff    : 9 originally-iff_omega_pair, now omega-minimal.
                       NEW to NS24's training (NS22 only saw them as
                       held-out cross-pool wins).
  - constructor      : Int.zero_le_ofNat, minimal `constructor <;> omega`
                       (in variant C only). NS22 trained it (wrongly) as
                       plain omega.

Held-out Int = Int theorems probed in the CX1/CX2 Int sets but not in
the 22-pool (includes the unresolved iff outlier Int.lt_toNat).

Verdicts:
  - memorization_only       : pool solved, no held-out gain vs NS22.
  - repaired_label_absorption: improves the relabeled_iff group over NS22
                               but no held-out gain (the repaired labels
                               were absorbed, nothing broader).
  - held_out_transfer       : adds held-out Int wins beyond NS22.
  - overfit_regression      : loses previously-solved Int or demo/Set/Finset.

Outputs:
  project/data/ns24_transfer_analysis.json
  project/evolve/reports/ns24_transfer_analysis.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

POOL_META = Path("project/data/ns23_minimal_family_pools_meta.json")

INT_SETS = [
    "cx2_int_iff_omega_easy",
    "cx2_int_iff_omega_medium",
    "cx2_int_order_arith",
    "cx2_int_mixed",
    "cx1_bool_option_int",
]
NEG_SETS = ["demo_v1", "ns17_set_extra", "ns17_finset_extra",
            "ns14_set_finset_extra"]

NS22_BASELINE = "gen_v5_ns22_int_fallback_omega_5x"

# Candidate NS24 checkpoints (only those present on disk are analyzed).
ALL_CANDIDATES = [
    "gen_v5_ns24_int_minimal_omega_5x",
    "gen_v5_ns24_int_minimal_omega_10x",
    "gen_v5_ns24_int_minimal_omega_plus_constructor_5x",
    "gen_v5_ns24_int_minimal_omega_5x_from_ns12",
]


def load_pool_groups() -> dict[str, dict]:
    meta = json.load(open(POOL_META))
    pool = meta["omega_aggregate_by_namespace"]["Int"]["theorems"]
    groups = {"old_ns22_omega": [], "relabeled_iff": [], "constructor": []}
    for e in pool:
        thm = e["theorem"]
        if e["minimal_tactic"].startswith("constructor"):
            groups["constructor"].append(thm)
        elif e["original_family"] == "iff_omega_pair":
            groups["relabeled_iff"].append(thm)
        else:
            groups["old_ns22_omega"].append(thm)
    return groups


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def per_thm(tag_prefix: str, ckpt: str, set_name: str) -> dict[str, dict]:
    """Read per-theorem rows for a (tag_prefix, ckpt, set) eval run."""
    p = first_match(
        f"project/evolve/eval_runs/{tag_prefix}_{ckpt}_{set_name}/"
        "eval-*/metrics.json"
    )
    if not p and ckpt == "gen_v5_ns12_balanced":
        for alt in [
            f"project/evolve/eval_runs/cx2_raw_{set_name}/eval-*/metrics.json",
            f"project/evolve/eval_runs/cx1_raw_{set_name}/eval-*/metrics.json",
            f"project/evolve/eval_runs/ns21_rawckpt_gen_v5_ns12_balanced_"
            f"{set_name}/eval-*/metrics.json",
        ]:
            p = first_match(alt)
            if p:
                break
    if not p:
        return {}
    return {t["full_name"]: t for t in
            json.load(open(p)).get("per_theorem", [])}


def collect_int(tag_prefix: str, ckpt: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for s in INT_SETS:
        for thm, blob in per_thm(tag_prefix, ckpt, s).items():
            if thm not in out:
                out[thm] = {**blob, "_set": s}
    return out


def solved(blob: dict) -> bool:
    return bool(blob.get("finished"))


def tac_of(blob: dict) -> str:
    return (blob.get("winning_tactic") or "")[:80]


def analyze(candidate: str, groups: dict) -> dict:
    cand = collect_int("ns24_rawckpt", candidate)
    ns22 = collect_int("ns22_rawckpt", NS22_BASELINE)
    ns12 = collect_int("ns22_rawckpt", "gen_v5_ns12_balanced")

    pool_all = set(sum(groups.values(), []))
    res: dict = {"candidate": candidate, "groups": {}}

    for gname, thms in groups.items():
        detail = []
        n_cand = n_ns22 = n_ns12 = 0
        for thm in thms:
            c = solved(cand.get(thm, {}))
            b = solved(ns22.get(thm, {}))
            z = solved(ns12.get(thm, {}))
            n_cand += c
            n_ns22 += b
            n_ns12 += z
            detail.append({
                "theorem": thm,
                "ns12_solved": z, "ns22_solved": b, "candidate_solved": c,
                "candidate_tactic": tac_of(cand.get(thm, {})),
            })
        res["groups"][gname] = {
            "size": len(thms),
            "ns12_solved": n_ns12,
            "ns22_solved": n_ns22,
            "candidate_solved": n_cand,
            "detail": detail,
        }

    # Held-out Int (probed Int theorems not in the pool).
    held = (set(cand) | set(ns22)) - pool_all
    gains, losses = [], []
    for thm in held:
        c = solved(cand.get(thm, {}))
        b = solved(ns22.get(thm, {}))
        if c and not b:
            gains.append({"theorem": thm,
                          "tactic": tac_of(cand.get(thm, {})),
                          "set": cand.get(thm, {}).get("_set")})
        elif b and not c:
            losses.append({"theorem": thm,
                           "set": ns22.get(thm, {}).get("_set")})
    res["held_out_int"] = {
        "n_probed": len(held),
        "ns22_wins": sum(1 for t in held if solved(ns22.get(t, {}))),
        "candidate_wins": sum(1 for t in held if solved(cand.get(t, {}))),
        "n_gains_vs_ns22": len(gains),
        "n_losses_vs_ns22": len(losses),
        "gains": sorted(gains, key=lambda g: g["theorem"]),
        "losses": sorted(losses, key=lambda g: g["theorem"]),
    }

    # Int totals across all sets.
    int_all = set(cand) | set(ns22) | set(ns12)
    res["int_totals"] = {
        "ns12": sum(1 for t in int_all if solved(ns12.get(t, {}))),
        "ns22": sum(1 for t in int_all if solved(ns22.get(t, {}))),
        "candidate": sum(1 for t in int_all if solved(cand.get(t, {}))),
        "n_int_probed": len(int_all),
    }

    # Emitted tactic distribution on solved Int theorems.
    tac_dist: dict[str, int] = {}
    for thm, blob in cand.items():
        if solved(blob):
            t = (blob.get("winning_tactic") or "").strip()
            key = ("omega" if t == "omega"
                   else "constructor<;>omega" if t.startswith("constructor")
                   else "aesop" if t == "aesop"
                   else "simp_all" if t.startswith("simp")
                   else "iff_pair" if "fun h => by omega" in t
                   else "other")
            tac_dist[key] = tac_dist.get(key, 0) + 1
    res["candidate_tactic_distribution"] = tac_dist

    # Negative control. NOTE: in the router, Set goals → ns12 and Finset
    # → ns21, never to the Int checkpoint, so the Int checkpoint's raw
    # Set/Finset score is NOT a production signal. Only demo_v1 (mixed,
    # partially Int-routed) is a meaningful regression signal here. We
    # also skip sets the candidate was never evaluated on (avoids a
    # false "loss" artifact from a missing eval run).
    neg = {}
    for s in NEG_SETS:
        cthms = per_thm("ns24_rawckpt", candidate, s)
        bthms = per_thm("ns22_rawckpt", NS22_BASELINE, s)
        if not cthms:
            neg[s] = {"evaluated": False, "ns22_wins": None,
                      "candidate_wins": None, "gains": None,
                      "losses": None, "loss_names": []}
            continue
        g = l = 0
        loss_names = []
        for thm in set(cthms) | set(bthms):
            c = solved(cthms.get(thm, {}))
            b = solved(bthms.get(thm, {}))
            if c and not b:
                g += 1
            elif b and not c:
                l += 1
                loss_names.append(thm)
        neg[s] = {
            "evaluated": True,
            "routed_away": s in ("ns17_set_extra", "ns17_finset_extra",
                                 "ns14_set_finset_extra"),
            "ns22_wins": sum(1 for t in set(cthms) | set(bthms)
                             if solved(bthms.get(t, {}))),
            "candidate_wins": sum(1 for t in set(cthms) | set(bthms)
                                  if solved(cthms.get(t, {}))),
            "gains": g, "losses": l, "loss_names": sorted(loss_names),
        }
    res["neg_control"] = neg

    # Verdict. Regression is judged on demo_v1 (the only routed-relevant
    # neg-control set) plus held-out Int losses; raw Set/Finset deltas are
    # routed away and excluded.
    relab = res["groups"]["relabeled_iff"]
    held_gains = res["held_out_int"]["n_gains_vs_ns22"]
    int_delta = res["int_totals"]["candidate"] - res["int_totals"]["ns22"]
    demo = neg.get("demo_v1", {})
    demo_losses = demo.get("losses") or 0
    int_losses = res["held_out_int"]["n_losses_vs_ns22"]
    neg_losses_routed = demo_losses  # routed-relevant regression signal

    if demo_losses >= 2 or int_losses >= 2:
        verdict = "regression"
    elif held_gains >= 3 and int_delta >= 3:
        verdict = "held_out_transfer"
    elif relab["candidate_solved"] > relab["ns22_solved"]:
        verdict = "repaired_label_absorption"
    elif int_delta >= 1:
        verdict = "marginal_gain"
    elif int_delta == 0 and held_gains <= 1:
        verdict = "reproduction_near_null"
    else:
        verdict = "marginal"

    res["summary"] = {
        "int_total_ns12": res["int_totals"]["ns12"],
        "int_total_ns22": res["int_totals"]["ns22"],
        "int_total_candidate": res["int_totals"]["candidate"],
        "int_delta_vs_ns22": int_delta,
        "relabeled_iff_ns22": relab["ns22_solved"],
        "relabeled_iff_candidate": relab["candidate_solved"],
        "held_out_gains_vs_ns22": held_gains,
        "held_out_losses_vs_ns22": int_losses,
        "demo_losses": demo_losses,
        "neg_control_losses_routed": neg_losses_routed,
        "verdict": verdict,
    }
    return res


def main() -> None:
    candidates = [c for c in (sys.argv[1:] or ALL_CANDIDATES)
                  if Path(f"project/models/{c}").is_dir()]
    if not candidates:
        raise SystemExit("no NS24 candidate checkpoints found on disk")
    groups = load_pool_groups()

    out = {"pool_groups": groups, "ns22_baseline": NS22_BASELINE,
           "candidates": {}}
    for c in candidates:
        out["candidates"][c] = analyze(c, groups)

    Path("project/data/ns24_transfer_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# NS24 — Int minimal-omega transfer vs absorption analysis", ""]
    md.append("Baseline = NS22 `gen_v5_ns22_int_fallback_omega_5x`. "
              "Pool groups under NS23 repaired labels: "
              f"old_ns22_omega={len(groups['old_ns22_omega'])}, "
              f"relabeled_iff={len(groups['relabeled_iff'])}, "
              f"constructor={len(groups['constructor'])}.")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| candidate | Int NS12 | Int NS22 | Int cand | Δ vs NS22 | "
              "relabeled_iff NS22→cand | held-out gains | held-out losses | "
              "demo losses | verdict |")
    md.append("|---|---:|---:|---:|---:|:---:|---:|---:|---:|---|")
    for c in candidates:
        s = out["candidates"][c]["summary"]
        md.append(
            f"| `{c}` | {s['int_total_ns12']} | {s['int_total_ns22']} | "
            f"**{s['int_total_candidate']}** | "
            f"{s['int_delta_vs_ns22']:+d} | "
            f"{s['relabeled_iff_ns22']}→{s['relabeled_iff_candidate']} | "
            f"{s['held_out_gains_vs_ns22']} | {s['held_out_losses_vs_ns22']} | "
            f"{s['demo_losses']} | **{s['verdict']}** |"
        )
    md.append("")

    for c in candidates:
        r = out["candidates"][c]
        md.append(f"## `{c}`")
        md.append("")
        md.append("### Pool-group resolution (NS12 / NS22 / candidate)")
        md.append("")
        md.append("| group | size | NS12 | NS22 | candidate |")
        md.append("|---|---:|---:|---:|---:|")
        for g, gv in r["groups"].items():
            md.append(f"| {g} | {gv['size']} | {gv['ns12_solved']} | "
                      f"{gv['ns22_solved']} | **{gv['candidate_solved']}** |")
        md.append("")
        md.append("### Per-theorem (relabeled_iff group — the NS24 test)")
        md.append("")
        md.append("| theorem | NS22 | candidate | candidate tactic |")
        md.append("|---|:---:|:---:|---|")
        for d in r["groups"]["relabeled_iff"]["detail"]:
            md.append(f"| `{d['theorem']}` | {'✓' if d['ns22_solved'] else '—'} "
                      f"| {'✓' if d['candidate_solved'] else '—'} | "
                      f"`{d['candidate_tactic']}` |")
        md.append("")
        h = r["held_out_int"]
        md.append("### Held-out Int (not in 22-pool)")
        md.append("")
        md.append(f"- probed: {h['n_probed']}; NS22 wins {h['ns22_wins']}, "
                  f"candidate wins {h['candidate_wins']}")
        md.append(f"- **gains vs NS22: {h['n_gains_vs_ns22']}**, "
                  f"losses: {h['n_losses_vs_ns22']}")
        if h["gains"]:
            md.append("")
            md.append("| gained theorem | tactic | set |")
            md.append("|---|---|---|")
            for g in h["gains"]:
                md.append(f"| `{g['theorem']}` | `{g['tactic']}` | "
                          f"{g.get('set','?')} |")
        if h["losses"]:
            md.append("")
            md.append("Losses: " + ", ".join(f"`{x['theorem']}`"
                                              for x in h["losses"]))
        md.append("")
        md.append(f"### Emitted tactic distribution (solved Int): "
                  f"{r['candidate_tactic_distribution']}")
        md.append("")
        md.append("### Negative control")
        md.append("")
        md.append("| set | routed away? | NS22 | candidate | gains | losses |")
        md.append("|---|:---:|---:|---:|---:|---:|")
        for s in NEG_SETS:
            v = r["neg_control"][s]
            if not v.get("evaluated"):
                md.append(f"| {s} | — | n/a | n/a | n/a | n/a |")
                continue
            ra = "yes" if v.get("routed_away") else "no"
            md.append(f"| {s} | {ra} | {v['ns22_wins']} | {v['candidate_wins']} | "
                      f"{v['gains']} | **{v['losses']}** |")
        md.append("")

    Path("project/evolve/reports/ns24_transfer_analysis.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print("wrote project/data/ns24_transfer_analysis.json")
    print("wrote project/evolve/reports/ns24_transfer_analysis.md")
    for c in candidates:
        s = out["candidates"][c]["summary"]
        print(f"  {c}: Int {s['int_total_ns22']}→{s['int_total_candidate']} "
              f"({s['int_delta_vs_ns22']:+d}); verdict={s['verdict']}")


if __name__ == "__main__":
    main()
