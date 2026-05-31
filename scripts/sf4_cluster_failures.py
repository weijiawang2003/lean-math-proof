#!/usr/bin/env python3
"""SF4 Part 4 — cluster confirmed RC2 failures.

Reads rc2_failure_confirmation.json, keeps CONFIRMED_RC2_FAILURE theorems, and
clusters by namespace x goal-shape x tactic-symptom x theorem-name features.
Emits per-cluster candidate-family directions, risk, and a recommendation.

Pure analysis (no Lean). Goal shape / symptom are inferred from the failing
theorem's name and (when available) its last goal / last error.
"""
from __future__ import annotations

import argparse
import json
import os

NAME_FEATURES = [
    ("ite/if", ["ite", "dite"]),
    ("iff", ["_iff", "iff_"]),
    ("inter/union/diff", ["inter", "union", "diff", "sdiff"]),
    ("compl", ["compl"]),
    ("toFinset", ["toFinset", "to_finset"]),
    ("card", ["card"]),
    ("map/filter", ["map", "filter", "image"]),
    ("subset", ["subset", "ssubset", "antitone", "monotone"]),
    ("disjoint", ["disjoint"]),
    ("singleton", ["singleton", "pair"]),
    ("powerset", ["powerset"]),
    ("nonempty", ["nonempty", "empty"]),
]


def _name_features(fn):
    low = fn.lower()
    feats = [label for label, keys in NAME_FEATURES if any(k.lower() in low for k in keys)]
    return feats or ["other"]


def _goal_shape(fn, goal, err):
    low = fn.lower()
    g = (goal or "")
    if "_iff" in low or "↔" in g or " iff " in (g or ""):
        return "iff"
    if "subset" in low or "⊆" in g or "ssubset" in low:
        return "subset"
    if "mem" in low or "∈" in g:
        return "membership"
    if "ext" in low:
        return "extensionality"
    if "card" in low or "coe" in low:
        return "coercion"
    if "=" in g or "_eq" in low or low.endswith("_eq") or "eq_" in low:
        return "equality"
    if any(k in low for k in ["add", "mul", "sub", "mod", "div", "le", "lt"]):
        return "arithmetic"
    return "unknown"


def _symptom(rec):
    err = (rec.get("error_message") or "").lower()
    used = rec.get("tactics_used") or []
    syms = []
    if "maximum recursion" in err or "maxrecdepth" in err:
        syms.append("max recursion")
    if "unexpected token" in err or "expected" in err:
        syms.append("parse issue")
    if "aesop" in used and not rec.get("rc2_finished"):
        syms.append("aesop failed")
    if any("simp" in t for t in used):
        syms.append("simp failed")
    if not syms:
        syms.append("missing bridge lemma likely")
    return syms


def _candidate_families(ns, shape, feats):
    fams = []
    if ns == "Set":
        if "ite/if" in feats:
            fams += ["set_ite_simp_aesop", "set_ite_ext"]
        if shape in ("equality", "extensionality") or "inter/union/diff" in feats or "compl" in feats:
            fams += ["set_ext_aesop", "set_ext_simp"]
        if shape == "iff":
            fams += ["set_iff_constructor_aesop"]
        if shape == "subset" or "subset" in feats:
            fams += ["set_subset_antisymm"]
        if "singleton" in feats or "powerset" in feats:
            fams += ["set_ext_aesop"]
    elif ns in ("Multiset", "Finset") or "toFinset" in feats:
        fams += ["multiset_tofinset_simp_aesop"]
    elif ns in ("Nat", "Int"):
        fams += ["arith_omega", "arith_nlinarith"]
    if not fams:
        fams = ["generic_aesop_simpall"]
    # dedupe preserving order
    seen, out = set(), []
    for f in fams:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def _risk_reco(ns, fams):
    broad = {"set_subset_antisymm", "arith_nlinarith", "generic_aesop_simpall"}
    risk = "high" if any(f in broad for f in fams) else ("low" if ns == "Set" else "medium")
    if any("simp_aesop" in f or "ext" in f or "ite" in f for f in fams):
        reco = "sequence_probe"
    elif any("omega" in f or "nlinarith" in f for f in fams):
        reco = "tactic_probe"
    else:
        reco = "sequence_probe"
    return risk, reco


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirmed", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    conf = json.load(open(args.confirmed))
    failures = [r for r in conf.get("results", []) if r.get("classification") == "CONFIRMED_RC2_FAILURE"]

    enriched = []
    for r in failures:
        fn = r["full_name"]
        ns = r.get("namespace") or (fn.split(".")[0] if "." in fn else "unknown")
        ns_bucket = ns if ns in ("Set", "Finset", "Multiset", "Nat", "List", "Algebra", "Order") else "unknown"
        feats = _name_features(fn)
        shape = _goal_shape(fn, r.get("last_goal"), r.get("error_message"))
        syms = _symptom(r)
        enriched.append({"full_name": fn, "file_path": r.get("file_path"), "namespace": ns,
                         "ns_bucket": ns_bucket, "name_features": feats, "goal_shape": shape,
                         "symptoms": syms})

    # cluster key = (ns_bucket, primary name feature, goal_shape)
    clusters = {}
    for e in enriched:
        key = (e["ns_bucket"], e["name_features"][0], e["goal_shape"])
        clusters.setdefault(key, []).append(e)

    out_clusters = []
    for (ns, feat, shape), members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        cid = f"{ns}__{feat.replace('/', '_')}__{shape}"
        fams = _candidate_families(ns, shape, [feat] + sorted({f for m in members for f in m["name_features"]}))
        risk, reco = _risk_reco(ns, fams)
        all_feats = sorted({f for m in members for f in m["name_features"]})
        all_syms = sorted({s for m in members for s in m["symptoms"]})
        out_clusters.append({
            "cluster_id": cid, "size": len(members),
            "namespace": ns, "primary_name_feature": feat, "goal_shape": shape,
            "representatives": [m["full_name"] for m in members[:5]],
            "members": [m["full_name"] for m in members],
            "common_goal_features": all_feats,
            "symptoms": all_syms,
            "candidate_families": fams,
            "risk": risk, "recommendation": reco,
        })

    out = {"confirmed_input": args.confirmed,
           "num_confirmed_failures": len(failures),
           "num_clusters": len(out_clusters),
           "clusters": out_clusters}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 RC2 failure clusters", "",
         f"- confirmed failures clustered: **{len(failures)}**",
         f"- clusters: **{len(out_clusters)}**", "",
         "| cluster_id | size | ns | shape | families | risk | reco |",
         "|---|---|---|---|---|---|---|"]
    for c in out_clusters:
        L.append(f"| `{c['cluster_id']}` | {c['size']} | {c['namespace']} | {c['goal_shape']} | "
                 f"{','.join(c['candidate_families'])} | {c['risk']} | {c['recommendation']} |")
    L += ["", "## Cluster detail", ""]
    for c in out_clusters:
        L.append(f"### `{c['cluster_id']}` (size {c['size']}, risk {c['risk']})")
        L.append(f"- representatives: {c['representatives']}")
        L.append(f"- common features: {c['common_goal_features']}")
        L.append(f"- symptoms: {c['symptoms']}")
        L.append(f"- candidate families: {c['candidate_families']}")
        L.append(f"- recommendation: **{c['recommendation']}**")
        L.append("")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-cluster] confirmed={len(failures)} clusters={len(out_clusters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
