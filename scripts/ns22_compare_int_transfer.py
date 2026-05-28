"""NS22 — transfer-vs-memorization analysis for Int family.

Compares each NS22 candidate (iff_5x, iff_10x, omega_5x) against the
NS12-balanced baseline on Int surfaces. Classifies by:

  - pool_solved: which trained pool theorems are solved raw, and what
    tactic the model emits (whether the trained pattern OR a substitute)
  - held_out_int_gains: new Int wins beyond the training pool
  - cross_family_gains: wins on the *other* family's pool (iff trained
    → omega pool solved, or vice versa) — evidence of broader transfer
  - regressions on Set/Finset/demo

Verdicts:
  - pool memorization (pool solved, ~0 held-out)
  - same-family transfer (pool + same-family held-out gains)
  - cross-family transfer (pool + other-family pool wins)
  - broader Int arithmetic improvement (gains beyond all pool surfaces)
  - regression-with-no-gain (pool not solved + neg-control losses)

Outputs:
  - project/data/ns22_transfer_analysis.json
  - project/evolve/reports/ns22_transfer_analysis.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


POOL_IFF = {
    "Int.le_add_one_iff", "Int.le_iff_lt_or_eq", "Int.le_sub_one_iff",
    "Int.sub_one_lt_iff", "Int.le_antisymm_iff", "Int.le_iff_eq_or_lt",
    "Int.natCast_nonpos_iff", "Int.natCast_ne_zero_iff_pos",
    "Int.lt_toNat", "Int.natCast_eq_zero",
}
POOL_OMEGA = {
    "Int.emod_two_eq_zero_or_one", "Int.le_of_eq",
    "Int.natAbs_coe_sub_coe_lt_of_lt", "Int.le_or_lt",
    "Int.natAbs_coe_sub_coe_le_of_le", "Int.zero_le_ofNat",
    "Int.lt_or_lt_of_ne", "Int.natAbs_add_of_nonpos", "Int.lt_asymm",
    "Int.le_natCast_sub", "Int.neg_emod_two", "Int.lt_or_le",
    "Int.natCast_pred_of_pos",
}
ALL_POOL = POOL_IFF | POOL_OMEGA

CANDIDATES = [
    "gen_v5_ns22_int_iff_omega_5x",
    "gen_v5_ns22_int_iff_omega_10x",
    "gen_v5_ns22_int_fallback_omega_5x",
]

CANDIDATE_OWN_POOL = {
    "gen_v5_ns22_int_iff_omega_5x": POOL_IFF,
    "gen_v5_ns22_int_iff_omega_10x": POOL_IFF,
    "gen_v5_ns22_int_fallback_omega_5x": POOL_OMEGA,
}

BASELINE = "gen_v5_ns12_balanced"

INT_SETS = [
    "cx2_int_iff_omega_easy",
    "cx2_int_iff_omega_medium",
    "cx2_int_order_arith",
    "cx2_int_mixed",
    "cx1_bool_option_int",
]
NEG_SETS = [
    "demo_v1", "ns17_set_extra",
    "ns17_finset_extra", "ns14_set_finset_extra",
]


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def per_thm_for(ckpt: str, set_name: str) -> dict[str, dict]:
    p = first_match(
        f"project/evolve/eval_runs/ns22_rawckpt_{ckpt}_{set_name}/"
        "eval-*/metrics.json"
    )
    if not p:
        # Fall back to canonical NS12 baseline tags.
        if ckpt == BASELINE:
            for alt in [
                f"project/evolve/eval_runs/cx2_raw_{set_name}/"
                "eval-*/metrics.json",
                f"project/evolve/eval_runs/cx1_raw_{set_name}/"
                "eval-*/metrics.json",
                f"project/evolve/eval_runs/ns21_rawckpt_{BASELINE}_"
                f"{set_name}/eval-*/metrics.json",
            ]:
                p = first_match(alt)
                if p:
                    break
    if not p:
        return {}
    return {t["full_name"]: t for t in
            json.load(open(p)).get("per_theorem", [])}


def analyze(ckpt: str) -> dict:
    own_pool = CANDIDATE_OWN_POOL[ckpt]
    other_pool = ALL_POOL - own_pool

    result: dict = {
        "ckpt": ckpt,
        "own_pool_family": ("iff_omega_pair"
                            if own_pool == POOL_IFF else "fallback_omega"),
        "pool_resolution": {},
        "cross_family_resolution": {},
        "held_out_int": {},
        "neg_control": {},
    }

    # Combine all int-set lookups into per-theorem candidate map.
    int_thms_cand: dict[str, dict] = {}
    int_thms_base: dict[str, dict] = {}
    for s in INT_SETS:
        for thm, blob in per_thm_for(ckpt, s).items():
            if thm not in int_thms_cand:
                int_thms_cand[thm] = {**blob, "_set": s}
        for thm, blob in per_thm_for(BASELINE, s).items():
            if thm not in int_thms_base:
                int_thms_base[thm] = {**blob, "_set": s}

    # Own pool resolution.
    own_solved = 0
    own_via_trained_tactic = 0
    pool_detail = []
    for thm in own_pool:
        blob = int_thms_cand.get(thm, {})
        solved = bool(blob.get("finished"))
        tac = (blob.get("winning_tactic") or blob.get("last_tactic") or "")
        if solved:
            own_solved += 1
            # Trained tactic for iff pool = iff-pair pattern
            # Trained tactic for omega pool = "omega"
            if own_pool == POOL_IFF:
                if "fun h => by omega" in tac and tac.count("by omega") >= 2:
                    own_via_trained_tactic += 1
            else:
                if tac.strip() == "omega":
                    own_via_trained_tactic += 1
        pool_detail.append({
            "theorem": thm, "solved": solved, "tactic": tac[:80],
            "set": blob.get("_set"),
        })
    result["pool_resolution"] = {
        "own_pool_solved": own_solved,
        "own_pool_size": len(own_pool),
        "own_via_trained_tactic": own_via_trained_tactic,
        "detail": pool_detail,
    }

    # Cross-family pool resolution (the *other* family's pool).
    cross_solved = 0
    cross_detail = []
    for thm in other_pool:
        blob = int_thms_cand.get(thm, {})
        solved = bool(blob.get("finished"))
        tac = (blob.get("winning_tactic") or blob.get("last_tactic") or "")
        if solved:
            cross_solved += 1
        cross_detail.append({
            "theorem": thm, "solved": solved, "tactic": tac[:80],
            "set": blob.get("_set"),
        })
    result["cross_family_resolution"] = {
        "other_pool_solved": cross_solved,
        "other_pool_size": len(other_pool),
        "detail": cross_detail,
    }

    # Held-out Int gains/losses (Int theorems not in any pool).
    gains = []
    losses = []
    held_out_int_thms = (
        (set(int_thms_cand) | set(int_thms_base)) - ALL_POOL
    )
    for thm in held_out_int_thms:
        c = bool(int_thms_cand.get(thm, {}).get("finished"))
        b = bool(int_thms_base.get(thm, {}).get("finished"))
        if c and not b:
            gains.append({
                "theorem": thm,
                "tactic": (int_thms_cand.get(thm, {}).get(
                    "winning_tactic") or "")[:60],
                "set": int_thms_cand.get(thm, {}).get("_set"),
            })
        elif b and not c:
            losses.append({"theorem": thm})
    result["held_out_int"] = {
        "n_int_held_out": len(held_out_int_thms),
        "n_baseline_wins": sum(
            1 for t in held_out_int_thms
            if int_thms_base.get(t, {}).get("finished")
        ),
        "n_candidate_wins": sum(
            1 for t in held_out_int_thms
            if int_thms_cand.get(t, {}).get("finished")
        ),
        "n_gains": len(gains),
        "n_losses": len(losses),
        "gains": gains[:25],
        "losses": losses,
    }

    # Negative control: demo/Set/Finset.
    for s in NEG_SETS:
        c_thms = per_thm_for(ckpt, s)
        b_thms = per_thm_for(BASELINE, s)
        gains_n = []
        losses_n = []
        for thm in set(c_thms) | set(b_thms):
            c = bool(c_thms.get(thm, {}).get("finished"))
            b = bool(b_thms.get(thm, {}).get("finished"))
            if c and not b:
                gains_n.append(thm)
            elif b and not c:
                losses_n.append(thm)
        result["neg_control"][s] = {
            "n_baseline_wins": sum(
                1 for t in set(c_thms) | set(b_thms)
                if b_thms.get(t, {}).get("finished")
            ),
            "n_candidate_wins": sum(
                1 for t in set(c_thms) | set(b_thms)
                if c_thms.get(t, {}).get("finished")
            ),
            "n_gains": len(gains_n),
            "n_losses": len(losses_n),
            "losses": sorted(losses_n),
        }

    # Verdict.
    n_own = result["pool_resolution"]["own_pool_solved"]
    n_cross = result["cross_family_resolution"]["other_pool_solved"]
    held_out_gains = result["held_out_int"]["n_gains"]
    neg_losses = sum(v["n_losses"] for v in result["neg_control"].values())

    if n_own == 0 and n_cross == 0 and held_out_gains == 0:
        verdict = "no_signal"
    elif n_own >= len(own_pool) * 0.7 and n_cross >= 3 and held_out_gains >= 3:
        verdict = "broad_transfer"
    elif n_own >= len(own_pool) * 0.7 and n_cross >= 3:
        verdict = "cross_family_transfer"
    elif n_own >= len(own_pool) * 0.7 and held_out_gains >= 1:
        verdict = "same_family_transfer"
    elif n_own >= len(own_pool) * 0.7:
        verdict = "pool_memorization"
    elif n_cross >= 3 or held_out_gains >= 3:
        verdict = "indirect_transfer"
    else:
        verdict = "weak_or_no_signal"
    if neg_losses >= 5:
        verdict = f"{verdict}_with_regression"

    result["summary"] = {
        "own_pool_solved": n_own,
        "own_pool_size": len(own_pool),
        "own_via_trained_tactic": result["pool_resolution"][
            "own_via_trained_tactic"],
        "other_pool_solved": n_cross,
        "other_pool_size": len(other_pool),
        "held_out_int_gains": held_out_gains,
        "neg_control_losses": neg_losses,
        "verdict": verdict,
    }
    return result


def main() -> None:
    out: dict = {
        "iff_omega_pool": sorted(POOL_IFF),
        "fallback_omega_pool": sorted(POOL_OMEGA),
        "candidates": {},
    }
    for c in CANDIDATES:
        out["candidates"][c] = analyze(c)
    Path("project/data/ns22_transfer_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    md = ["# NS22 — Int transfer vs memorization analysis", ""]
    md.append("## Summary")
    md.append("")
    md.append("| ckpt | own family | own pool | via trained tactic | "
              "other pool | held-out gains | neg losses | verdict |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for c in CANDIDATES:
        s = out["candidates"][c]["summary"]
        own_fam = out["candidates"][c]["own_pool_family"]
        md.append(
            f"| `{c}` | {own_fam} | "
            f"{s['own_pool_solved']}/{s['own_pool_size']} | "
            f"{s['own_via_trained_tactic']} | "
            f"{s['other_pool_solved']}/{s['other_pool_size']} | "
            f"{s['held_out_int_gains']} | "
            f"{s['neg_control_losses']} | **{s['verdict']}** |"
        )
    md.append("")

    md.append("## Own-pool detail")
    md.append("")
    for c in CANDIDATES:
        own_fam = out["candidates"][c]["own_pool_family"]
        md.append(f"### `{c}` (own pool = {own_fam})")
        md.append("")
        md.append("| theorem | solved | tactic emitted |")
        md.append("|---|:---:|---|")
        for d in out["candidates"][c]["pool_resolution"]["detail"]:
            sol = "✓" if d["solved"] else "—"
            md.append(f"| `{d['theorem']}` | {sol} | `{d['tactic']}` |")
        md.append("")

    md.append("## Cross-family pool resolution")
    md.append("")
    md.append("How many of the *other* pool's theorems each candidate "
              "solves. This is the cleanest evidence of broader transfer: "
              "training on family X solving family Y's theorems via the "
              "trained-family tactic (or any other shorter tactic).")
    md.append("")
    for c in CANDIDATES:
        cross = out["candidates"][c]["cross_family_resolution"]
        md.append(f"### `{c}`: {cross['other_pool_solved']}/"
                  f"{cross['other_pool_size']}")
        md.append("")
        if cross["other_pool_solved"]:
            md.append("| theorem | tactic |")
            md.append("|---|---|")
            for d in cross["detail"]:
                if d["solved"]:
                    md.append(f"| `{d['theorem']}` | `{d['tactic']}` |")
            md.append("")

    md.append("## Held-out Int gains (beyond all pool theorems)")
    md.append("")
    for c in CANDIDATES:
        h = out["candidates"][c]["held_out_int"]
        md.append(f"### `{c}`")
        md.append("")
        md.append(f"- held-out Int theorems probed: "
                  f"{h['n_int_held_out']}")
        md.append(f"- NS12 baseline wins: {h['n_baseline_wins']}")
        md.append(f"- candidate wins: {h['n_candidate_wins']}")
        md.append(f"- **gains: {h['n_gains']}**, losses: {h['n_losses']}")
        if h["gains"]:
            md.append("")
            md.append("Gain detail (first 25):")
            md.append("")
            md.append("| theorem | tactic | set |")
            md.append("|---|---|---|")
            for g in h["gains"]:
                md.append(f"| `{g['theorem']}` | `{g['tactic']}` | "
                          f"{g.get('set','?')} |")
        md.append("")

    md.append("## Negative control (Set/Finset/demo)")
    md.append("")
    for c in CANDIDATES:
        md.append(f"### `{c}`")
        md.append("")
        md.append("| set | baseline | candidate | gains | losses |")
        md.append("|---|---:|---:|---:|---:|")
        for s in NEG_SETS:
            v = out["candidates"][c]["neg_control"].get(s, {})
            md.append(
                f"| {s} | {v.get('n_baseline_wins', 0)} | "
                f"{v.get('n_candidate_wins', 0)} | "
                f"{v.get('n_gains', 0)} | "
                f"**{v.get('n_losses', 0)}** |"
            )
        md.append("")

    Path("project/evolve/reports/ns22_transfer_analysis.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("wrote project/data/ns22_transfer_analysis.json")
    print("wrote project/evolve/reports/ns22_transfer_analysis.md")
    for c in CANDIDATES:
        s = out["candidates"][c]["summary"]
        print(f"  {c}: verdict={s['verdict']}")
        print(f"    own_pool_solved={s['own_pool_solved']}/"
              f"{s['own_pool_size']} "
              f"(via trained tactic: {s['own_via_trained_tactic']})")
        print(f"    other_pool_solved={s['other_pool_solved']}/"
              f"{s['other_pool_size']}")
        print(f"    held_out_int_gains={s['held_out_int_gains']}")
        print(f"    neg_losses={s['neg_control_losses']}")


if __name__ == "__main__":
    main()
