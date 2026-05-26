"""NS21 — transfer-vs-memorization analysis.

For each NS21 candidate checkpoint and each Finset eval set:

  - pool theorems (the 6): was it solved by the candidate? what
    tactic did the model emit?
  - non-pool ("held-out") Finset theorems on the same set: did the
    candidate add or lose any wins vs gen_v5_ns12_balanced?

We classify each candidate as:
  - memorization: pool theorems solved, ~0 held-out gains.
  - narrow transfer: pool + 1-2 held-out gains.
  - broad transfer: pool + ≥3 held-out gains.
  - regression: net loss on Set/demo without held-out Finset gains.

Outputs:
  - project/data/ns21_transfer_analysis.json
  - project/evolve/reports/ns21_transfer_analysis.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


POOL_THEOREMS = {
    "Finset.coe_insert",
    "Finset.cons_eq_insert",
    "Finset.disjUnion_singleton",
    "Finset.coe_cons",
    "Finset.card_insert_eq_ite",
    "Finset.image_id",
}

CANDIDATES = [
    "gen_v5_ns21_finset_aesop_10x",
    "gen_v5_ns21_finset_aesop_20x",
    "gen_v5_ns21_finset_aesop_minimal",
]

BASELINE = "gen_v5_ns12_balanced"

FINSET_SETS = [
    "ns17_finset_extra",
    "cx1_finset_image_filter",
    "ns20_finset_aesop_extra_easy",
    "ns20_finset_aesop_extra_medium",
    "ns20_finset_aesop_extra_hard",
]

NEG_SETS = ["ns17_set_extra", "ns14_set_finset_extra", "demo_v1"]


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def per_thm(ckpt: str, set_name: str) -> dict[str, dict]:
    pat = (
        f"project/evolve/eval_runs/ns21_rawckpt_{ckpt}_{set_name}/"
        "eval-*/metrics.json"
    )
    p = first_match(pat)
    if not p:
        return {}
    return {t["full_name"]: t for t in json.loads(
        Path(p).read_text(encoding="utf-8")).get("per_theorem", [])}


def solved_with_tactic(blob: dict) -> tuple[bool, str]:
    if not blob.get("finished"):
        return False, blob.get("winning_tactic") or ""
    return True, blob.get("winning_tactic") or blob.get("last_tactic") or ""


def analyze_candidate(ckpt: str) -> dict:
    result: dict = {
        "ckpt": ckpt,
        "pool": {},
        "held_out_finset": {},
        "neg_control": {},
    }

    # Pool — which of the 6 are now solved raw?
    for s in FINSET_SETS:
        cand_thms = per_thm(ckpt, s)
        for thm in POOL_THEOREMS:
            if thm in cand_thms:
                solved, tac = solved_with_tactic(cand_thms[thm])
                # First-seen wins (only count it once per theorem).
                if thm not in result["pool"]:
                    result["pool"][thm] = {
                        "found_in_set": s,
                        "solved": solved,
                        "tactic": tac,
                    }

    # Held-out Finset — diff vs baseline on non-pool theorems.
    for s in FINSET_SETS:
        cand_thms = per_thm(ckpt, s)
        base_thms = per_thm(BASELINE, s)
        held_out = (set(cand_thms) | set(base_thms)) - POOL_THEOREMS
        gains = []
        losses = []
        for thm in held_out:
            c_blob = cand_thms.get(thm, {})
            b_blob = base_thms.get(thm, {})
            c_solved = bool(c_blob.get("finished"))
            b_solved = bool(b_blob.get("finished"))
            if c_solved and not b_solved:
                gains.append({
                    "theorem": thm,
                    "tactic": (c_blob.get("winning_tactic")
                               or c_blob.get("last_tactic") or ""),
                })
            elif b_solved and not c_solved:
                losses.append({"theorem": thm})
        result["held_out_finset"][s] = {
            "n_held_out_thms": len(held_out),
            "n_baseline_wins": sum(
                1 for t in held_out if base_thms.get(t, {}).get("finished")
            ),
            "n_candidate_wins": sum(
                1 for t in held_out if cand_thms.get(t, {}).get("finished")
            ),
            "n_gains": len(gains),
            "n_losses": len(losses),
            "gains": gains,
            "losses": losses,
        }

    # Negative control — Set/demo regressions.
    for s in NEG_SETS:
        cand_thms = per_thm(ckpt, s)
        base_thms = per_thm(BASELINE, s)
        gains = []
        losses = []
        for thm in set(cand_thms) | set(base_thms):
            c_solved = bool(cand_thms.get(thm, {}).get("finished"))
            b_solved = bool(base_thms.get(thm, {}).get("finished"))
            if c_solved and not b_solved:
                gains.append(thm)
            elif b_solved and not c_solved:
                losses.append(thm)
        result["neg_control"][s] = {
            "n_baseline_wins": sum(
                1 for t in set(cand_thms) | set(base_thms)
                if base_thms.get(t, {}).get("finished")
            ),
            "n_candidate_wins": sum(
                1 for t in set(cand_thms) | set(base_thms)
                if cand_thms.get(t, {}).get("finished")
            ),
            "n_gains": len(gains),
            "n_losses": len(losses),
            "losses": sorted(losses),
        }

    # Classify.
    pool_solved = sum(1 for v in result["pool"].values() if v["solved"])
    held_out_gains = sum(
        v["n_gains"] for v in result["held_out_finset"].values()
    )
    neg_losses = sum(
        v["n_losses"] for v in result["neg_control"].values()
    )
    if pool_solved >= 4 and held_out_gains >= 3:
        verdict = "broad_transfer"
    elif pool_solved >= 4 and held_out_gains >= 1:
        verdict = "narrow_transfer"
    elif pool_solved >= 4:
        verdict = "memorization"
    elif pool_solved >= 1:
        verdict = "partial_memorization"
    else:
        verdict = "no_signal"
    if neg_losses >= 5:
        verdict = f"{verdict}_with_regression"

    result["summary"] = {
        "pool_solved": pool_solved,
        "pool_total": len(POOL_THEOREMS),
        "held_out_finset_gains_total": held_out_gains,
        "neg_control_losses_total": neg_losses,
        "verdict": verdict,
    }
    return result


def main() -> None:
    out: dict = {"candidates": {}, "pool_theorems": sorted(POOL_THEOREMS)}
    for c in CANDIDATES:
        out["candidates"][c] = analyze_candidate(c)
    out_path = Path("project/data/ns21_transfer_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # Render markdown summary.
    md = ["# NS21 — transfer vs memorization analysis", ""]
    md.append("## Summary")
    md.append("")
    md.append("| ckpt | pool solved | held-out Finset gains | neg-control losses | verdict |")
    md.append("|---|---:|---:|---:|---|")
    for c in CANDIDATES:
        s = out["candidates"][c]["summary"]
        md.append(
            f"| `{c}` | {s['pool_solved']}/{s['pool_total']} | "
            f"{s['held_out_finset_gains_total']} | "
            f"{s['neg_control_losses_total']} | **{s['verdict']}** |"
        )
    md.append("")
    md.append("## Pool detail")
    md.append("")
    md.append("Which of the 6 training-pool theorems each candidate solves raw, "
              "and what tactic it emits.")
    md.append("")
    for c in CANDIDATES:
        md.append(f"### `{c}`")
        md.append("")
        md.append("| theorem | solved? | tactic |")
        md.append("|---|:---:|---|")
        for thm in sorted(POOL_THEOREMS):
            info = out["candidates"][c]["pool"].get(thm, {})
            solved = "✓" if info.get("solved") else "—"
            tac = (info.get("tactic") or "")[:60]
            md.append(f"| `{thm}` | {solved} | `{tac}` |")
        md.append("")

    md.append("## Held-out Finset transfer")
    md.append("")
    for c in CANDIDATES:
        md.append(f"### `{c}`")
        md.append("")
        md.append("| set | held-out | baseline wins | candidate wins | gains | losses |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for s in FINSET_SETS:
            v = out["candidates"][c]["held_out_finset"].get(s, {})
            md.append(
                f"| {s} | {v.get('n_held_out_thms', 0)} | "
                f"{v.get('n_baseline_wins', 0)} | "
                f"{v.get('n_candidate_wins', 0)} | "
                f"**{v.get('n_gains', 0)}** | {v.get('n_losses', 0)} |"
            )
        # List held-out gains if any.
        for s in FINSET_SETS:
            v = out["candidates"][c]["held_out_finset"].get(s, {})
            if v.get("n_gains"):
                md.append("")
                md.append(f"**gains on {s}**:")
                for g in v["gains"]:
                    md.append(f"- `{g['theorem']}` via `{g['tactic'][:60]}`")
        md.append("")

    md.append("## Negative control (Set/demo)")
    md.append("")
    for c in CANDIDATES:
        md.append(f"### `{c}`")
        md.append("")
        md.append("| set | baseline wins | candidate wins | gains | losses |")
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

    rep_path = Path("project/evolve/reports/ns21_transfer_analysis.md")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {rep_path}")


if __name__ == "__main__":
    main()
