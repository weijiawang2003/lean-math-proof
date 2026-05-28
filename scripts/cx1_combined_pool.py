"""CX1 — combined Finset/aesop and other-family pool meta across
NS18 + NS19 + NS20 + CX1.

Writes project/data/cx1_combined_pool_meta.json with per-family,
per-namespace unique wrapper-only-vs-NS9 counts. This is the
input for the NS21 training-gate decision.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def load_solved(p: str | None) -> set[str]:
    if not p:
        return set()
    return {t["full_name"] for t in
            json.loads(Path(p).read_text(encoding="utf-8")).get("per_theorem", [])
            if t.get("finished")}


def per_thm(p: str | None) -> dict[str, dict]:
    if not p:
        return {}
    return {t["full_name"]: t for t in
            json.loads(Path(p).read_text(encoding="utf-8")).get("per_theorem", [])}


def fam(t: str) -> str:
    import re
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t: return "empty"
    if t == "omega": return "fallback_omega"
    if t == "aesop": return "aesop"
    if t == "decide": return "fallback_decide"
    if t == "rfl": return "fallback_rfl"
    if t.startswith("constructor") and "omega" in t: return "constructor_omega"
    if t.startswith("split_ifs"): return "split_ifs"
    if "fun h => by omega" in t and t.count("by omega") >= 2: return "iff_omega_pair"
    if t.startswith("simp_all"): return "simp_all"
    if t.startswith("simp"): return "simp_other"
    if t.startswith("rw"): return "rw_named"
    if t.startswith("exact"): return "exact_named"
    if t.startswith("apply"): return "apply_named"
    return "other"


# (arc, variant_eval_dir_pattern, set_name, raw_pattern, wrap_pattern)
SOURCES = [
    # NS18 — aesop_wrapper on ns17 surfaces
    ("NS18", "ns18_aesop_wrapper_wrapper_{}",
     "ns17_finset_extra",
     "ns17_ns15routed_raw_ns17_finset_extra",
     "ns17_ns15routed_wrapper_ns17_finset_extra"),
    ("NS18", "ns18_aesop_wrapper_wrapper_{}",
     "ns17_set_extra",
     "ns17_ns15routed_raw_ns17_set_extra",
     "ns17_ns15routed_wrapper_ns17_set_extra"),
    # NS18 — nat_simp_arith on the div_mod surface
    ("NS18", "ns18_nat_simp_arith_wrapper_{}",
     "ns16_nat_div_mod_extra",
     "ns16_ns15routed_raw_ns16_nat_div_mod_extra",
     "ns16_ns15routed_wrapper_ns16_nat_div_mod_extra"),
    # NS18 — combined_safe on nat_defs_large_v5 (Nat.mod_mul_mod win)
    ("NS18", "ns18_combined_safe_wrapper_{}",
     "nat_defs_large_v5",
     "gen_v5_ns15_routed_raw_nat_defs_large_v5",
     "gen_v5_ns15_routed_wrapper_nat_defs_large_v5"),
    # NS19 — finset_aesop_only on the NS19 Finset surface
    ("NS19", "ns19_finset_aesop_only_wrapper_{}",
     "ns19_finset_aesop_surface",
     "ns19_raw_ns19_finset_aesop_surface",
     "ns19_ns9wrap_ns19_finset_aesop_surface"),
    # NS20 — finset_aesop_only across the three remainder surfaces
    ("NS20", "ns20_finset_aesop_only_wrapper_{}",
     "ns20_finset_aesop_extra_easy",
     "ns20_raw_ns20_finset_aesop_extra_easy",
     "ns20_ns9wrap_ns20_finset_aesop_extra_easy"),
    ("NS20", "ns20_finset_aesop_only_wrapper_{}",
     "ns20_finset_aesop_extra_medium",
     "ns20_raw_ns20_finset_aesop_extra_medium",
     "ns20_ns9wrap_ns20_finset_aesop_extra_medium"),
    ("NS20", "ns20_finset_aesop_only_wrapper_{}",
     "ns20_finset_aesop_extra_hard",
     "ns20_raw_ns20_finset_aesop_extra_hard",
     "ns20_ns9wrap_ns20_finset_aesop_extra_hard"),
    # CX1 — finset_aesop_only on cx1_finset_image_filter
    ("CX1", "cx1_finset_aesop_only_wrapper_{}",
     "cx1_finset_image_filter",
     "cx1_raw_cx1_finset_image_filter",
     "cx1_ns9wrap_cx1_finset_image_filter"),
    # CX1 — NS9 wrap on the four CX1 surfaces (wrap_solved − raw_solved
    # is the "wrapper-only" signal even without a variant)
    ("CX1", "cx1_ns9wrap_{}",
     "cx1_finset_image_filter",
     "cx1_raw_cx1_finset_image_filter",
     "cx1_ns9wrap_cx1_finset_image_filter"),
    ("CX1", "cx1_ns9wrap_{}",
     "cx1_nat_gcd_dvd_mod",
     "cx1_raw_cx1_nat_gcd_dvd_mod",
     "cx1_ns9wrap_cx1_nat_gcd_dvd_mod"),
    ("CX1", "cx1_ns9wrap_{}",
     "cx1_bool_option_int",
     "cx1_raw_cx1_bool_option_int",
     "cx1_ns9wrap_cx1_bool_option_int"),
    ("CX1", "cx1_ns9wrap_{}",
     "cx1_list_multiset",
     "cx1_raw_cx1_list_multiset",
     "cx1_ns9wrap_cx1_list_multiset"),
]


def main() -> None:
    pool: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"thms": {}, "sources": defaultdict(int)}
    )
    for arc, var_tpl, s, raw_pat, wrap_pat in SOURCES:
        vp = first_match(
            f"project/evolve/eval_runs/{var_tpl.format(s)}/eval-*/metrics.json"
        )
        rp = first_match(f"project/evolve/eval_runs/{raw_pat}/eval-*/metrics.json")
        wp = first_match(f"project/evolve/eval_runs/{wrap_pat}/eval-*/metrics.json")
        if not vp:
            continue
        vt = per_thm(vp)
        v_solved = {n for n, t in vt.items() if t.get("finished")}
        raw_solved = load_solved(rp)
        wrap_solved = load_solved(wp)
        # truly-new = wrapper-only-vs-raw. When the variant is an
        # experimental wrapper (e.g. ns18_aesop_wrapper,
        # ns19_finset_aesop_only), truly-new means variant proved AND
        # neither raw NS15 routed nor NS9 wrap baseline proved (so the
        # variant's contribution is uniquely its own). When the variant
        # IS the NS9 wrap itself (no experimental wrapper), truly-new
        # simply means wrap proved AND raw did not.
        if vp == wp:
            truly_new = v_solved - raw_solved
        else:
            truly_new = (v_solved - wrap_solved) - raw_solved
        for thm in truly_new:
            blob = vt.get(thm) or {}
            tac = (blob.get("winning_tactic")
                   or blob.get("last_tactic")
                   or blob.get("tactic") or "")
            f = fam(tac)
            ns = thm.split(".", 1)[0] if "." in thm else "?"
            key = (f, ns)
            if thm not in pool[key]["thms"]:
                pool[key]["thms"][thm] = {
                    "winning_tactic": tac,
                    "first_seen_in_arc": arc,
                    "first_seen_in_set": s,
                }
            pool[key]["sources"][f"{arc}:{s}"] += 1

    out: dict = {
        "training_gate_unique_required": 5,
        "families": {},
    }
    for (f, ns), info in pool.items():
        unique = len(info["thms"])
        if unique <= 1: osf = 20
        elif unique <= 3: osf = 15
        elif unique <= 6: osf = 10
        elif unique <= 12: osf = 5
        else: osf = 2
        out["families"][f"{f}|{ns}"] = {
            "family": f,
            "namespace": ns,
            "unique_count": unique,
            "trainable": unique >= 5,
            "recommended_oversample_factor": osf,
            "theorems": info["thms"],
            "source_breakdown": dict(info["sources"]),
        }
    # Sort by unique_count desc.
    out["families"] = dict(sorted(
        out["families"].items(),
        key=lambda kv: -kv[1]["unique_count"]
    ))
    out_path = Path("project/data/cx1_combined_pool_meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"\nCombined wrapper-only pool (NS18 + NS19 + NS20 + CX1):")
    for k, info in out["families"].items():
        gate = "TRAIN" if info["trainable"] else "  -- "
        print(f"  [{gate}] {k}: {info['unique_count']} unique")
        for thm, m in list(info["theorems"].items())[:10]:
            print(f"           {thm} | {m['winning_tactic'][:60]} | {m['first_seen_in_arc']}")


if __name__ == "__main__":
    main()
