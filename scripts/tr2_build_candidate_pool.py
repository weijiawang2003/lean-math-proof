#!/usr/bin/env python3
"""TR2 Part 2 — build the active-learning candidate pool.

Unions the TR1 active-learning / next-work / RC2-failure-prediction artifacts, the
SF4 failure pool / confirmation / clusters / missing-lemma triage, and the SF1
frontier; deduplicates by full_name; and tags each row with the TR1 router's
predictions (entropy / margin), its known literal-RC2 status (from SF4), its SF4
cluster, coarse goal features, and selection tags.

Every candidate is also flagged `in_tr1_training` — for TR2 the pool is fully
overlapping with the 57 TR1 examples, which is itself the central finding (the
fresh frontier is exhausted). Such rows are KEPT (as controls / re-probe targets)
but tagged, per the spec's "exclude already-in-TR1 unless needed as controls".

Pure analysis; no Lean. Router scoring uses the persisted TR1 vectorizers/model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

# ---- input defaults --------------------------------------------------------
TR1_AL = "project/evolve/experiments/tr1/out/tr1_active_learning_cases.json"
TR1_QUEUE = "project/evolve/experiments/tr1/out/tr1_next_work_queue.json"
TR1_PRED = "project/evolve/experiments/tr1/out/tr1_rc2_failure_predictions.json"
TR1_EX = "project/evolve/experiments/tr1/data/tr1_examples.jsonl"
SF4_POOL = "project/evolve/experiments/sf4/cases/rc2_failure_pool.jsonl"
SF4_CONF = "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json"
SF4_CLUST = "project/evolve/experiments/sf4/out/rc2_failure_clusters.json"
SF4_MISS = "project/evolve/experiments/sf4/out/sf4_missing_lemma_candidates.json"
FRONT_PATHS = "project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl"
FRONT_CLS = "project/evolve/experiments/sf1/out/real/classified_frontier.jsonl"

UNDERREP_LABELS = {"WX3_MULTISET_INDUCTION", "MX2_TOFINSET_AESOP", "PROOF_SEARCH_DEPTH_GAP"}


def _jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def _jd(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else (default or {})


def _entropy(dist):
    return float(-sum(q * math.log(q + 1e-12) for q in dist if q > 0))


def _goal_features(name, feats):
    """Coarse symbolic goal features from name + (optional) tr1 boolean flags."""
    n = name.lower()
    out = []
    F = feats or {}
    pairs = [("set", "has_set", ["set."]), ("ite", "has_ite", ["ite", "dite", ".if"]),
             ("iff", "has_iff", ["iff"]), ("subset", "has_subset", ["subset", "ssubset"]),
             ("multiset", "has_multiset", ["multiset"]), ("tofinset", "has_tofinset", ["tofinset"]),
             ("card", "has_card", ["card"]), ("singleton", "has_singleton", ["singleton"]),
             ("ext", "has_extensionality_shape", []), ("induction", "has_induction_signal", [])]
    for tag, fk, subs in pairs:
        if F.get(fk) or any(s in n for s in subs):
            out.append(tag)
    return sorted(set(out))


class _Router:
    """Lazy TR1 router for fresh in-sample scoring of pool members missing a
    held-out TR1 prediction (chiefly the RC2-solved cases)."""

    def __init__(self):
        self.ok = False
        try:
            import numpy as np
            import scipy.sparse as sp
            from scipy.sparse import load_npz
            import joblib
            from sklearn.linear_model import SGDClassifier
            meta = json.load(open("project/evolve/experiments/tr1/data/tr1_feature_metadata.json"))
            lm = json.load(open("project/evolve/experiments/tr1/data/tr1_label_map.json"))
            self.idx_to_label = {int(k): v for k, v in lm["index_to_label"].items()}
            X = load_npz("project/evolve/experiments/tr1/data/tr1_features.npz")
            yg = np.load(meta["yg_path"], allow_pickle=True)
            self.model = SGDClassifier(loss="log_loss", max_iter=3000,
                                       class_weight="balanced", alpha=1e-4, random_state=0)
            self.model.fit(X, yg["y"])
            self.vecs = joblib.load(meta["vectorizers_path"])
            self.np, self.sp = np, sp
            self.ok = True
        except Exception as e:  # router optional; pool still builds without it
            self.err = f"{type(e).__name__}: {e}"

    def score(self, fn):
        if not self.ok:
            return None
        np, sp = self.np, self.sp
        toks = " ".join(re.split(r"[._]", fn)).lower()
        blocks = [self.vecs["name_char"].transform([fn]), self.vecs["name_tok"].transform([toks]),
                  self.vecs["ns"].transform([{f"ns={fn.split('.')[0]}": 1.0}]),
                  self.vecs["goal_word"].transform(["∅"]), self.vecs["goal_char"].transform(["∅"]),
                  self.vecs["bool"].transform([{}]), self.vecs["cluster"].transform([{}])]
        Xr = sp.hstack(blocks).tocsr()
        full = np.zeros(len(self.idx_to_label))
        pr = self.model.predict_proba(Xr)[0]
        for j, cls in enumerate(self.model.classes_):
            full[cls] = pr[j]
        order = np.argsort(-full)
        top = [{"label": self.idx_to_label[int(j)], "score": round(float(full[j]), 3)}
               for j in order[:3]]
        return top, "tr1_full_model_in_sample"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-pool", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--out-summary-md", required=True)
    args = p.parse_args(argv)

    examples = _jl(TR1_EX)
    ex_by_name = {e["full_name"]: e for e in examples}
    tr1_names = set(ex_by_name)

    conf = _jd(SF4_CONF)
    conf_by_name = {r["full_name"]: r for r in conf.get("results", [])}

    # cluster membership
    clust = _jd(SF4_CLUST)
    cluster_of, cluster_meta = {}, {}
    for c in clust.get("clusters", []):
        cluster_meta[c["cluster_id"]] = c
        for m in c.get("members", []):
            cluster_of[m] = c["cluster_id"]

    # missing-lemma triage per cluster
    miss = _jd(SF4_MISS)
    triage_of = {t["cluster_id"]: t.get("category") for t in miss.get("triage", [])}

    # held-out TR1 predictions for the 27 confirmed failures
    pred = _jd(TR1_PRED)
    heldout = {r["full_name"]: r for r in pred.get("predictions", [])}
    # active-learning entropy/predicted_label for 25 cases
    al = _jd(TR1_AL)
    al_of = {c["full_name"]: c for c in al.get("selected", [])}

    router = _Router()

    # ---- gather all candidate names + their primary source ----
    sources = {}  # full_name -> ordered set of source tags

    def add_src(fn, tag):
        sources.setdefault(fn, [])
        if tag not in sources[fn]:
            sources[fn].append(tag)

    file_path_of, ns_of = {}, {}

    def record_meta(fn, fp, ns):
        if fp and fn not in file_path_of:
            file_path_of[fn] = fp
        if ns and fn not in ns_of:
            ns_of[fn] = ns

    for r in conf.get("results", []):
        add_src(r["full_name"], "sf4_confirmation")
        record_meta(r["full_name"], r.get("file_path"), r.get("namespace"))
    for r in _jl(SF4_POOL):
        add_src(r["full_name"], "sf4_failure_pool")
        record_meta(r["full_name"], r.get("file_path"), r.get("namespace"))
    for c in al.get("selected", []):
        add_src(c["full_name"], "tr1_active_learning")
        record_meta(c["full_name"], c.get("file_path"), c.get("namespace"))
    for q in _jd(TR1_QUEUE).get("queue", []):
        add_src(q["full_name"], "tr1_next_work_queue")
        record_meta(q["full_name"], q.get("file_path"), None)
    for r in heldout.values():
        add_src(r["full_name"], "tr1_rc2_predictions")
        record_meta(r["full_name"], r.get("file_path"), None)
    fr_cls = {r["decl_name"]: r for r in _jl(FRONT_CLS)}
    for r in _jl(FRONT_PATHS):
        fn = r.get("name") or r.get("full_name")
        add_src(fn, "sf1_frontier")
        record_meta(fn, r.get("file_path"), r.get("namespace"))

    rows = []
    excluded = []
    for fn in sorted(sources):
        fp = file_path_of.get(fn)
        ns = ns_of.get(fn) or (fn.split(".")[0] if "." in fn else "")
        ex = ex_by_name.get(fn)
        in_tr1 = fn in tr1_names
        # known RC2 status
        crec = conf_by_name.get(fn)
        if crec:
            cls = crec.get("classification")
            status = {"CONFIRMED_RC2_FAILURE": "failed", "NOW_SOLVED_BY_RC2": "solved"}.get(cls, "unknown")
        elif ex and ex.get("rc2_status") in ("solved", "failed"):
            status = ex["rc2_status"]
        else:
            status = "unknown"

        # router predictions
        pred_source, top_preds = None, None
        if fn in heldout:
            top_preds = heldout[fn].get("top_predictions")
            pred_source = "tr1_heldout"
        if not top_preds and fn in al_of:
            top_preds = [{"label": al_of[fn]["predicted_label"], "score": None}]
            pred_source = "tr1_active_learning"
        if not top_preds and router.ok:
            top_preds, pred_source = router.score(fn)
        # entropy / margin
        entropy = margin = None
        if top_preds and all(tp.get("score") is not None for tp in top_preds):
            scores = [tp["score"] for tp in top_preds]
            entropy = round(_entropy(scores + [max(0.0, 1.0 - sum(scores))]), 4)
            margin = round(scores[0] - (scores[1] if len(scores) > 1 else 0.0), 4)
        elif fn in al_of and al_of[fn].get("entropy") is not None:
            entropy = al_of[fn]["entropy"]

        cid = cluster_of.get(fn)
        goal_features = _goal_features(fn, (ex or {}).get("features"))
        pred_label = top_preds[0]["label"] if top_preds else None

        # selection tags
        tags = []
        if status == "failed":
            tags.append("confirmed_rc2_failure")
        elif status == "solved":
            tags.append("rc2_solved")
        if pred_label in UNDERREP_LABELS:
            tags.append("underrepresented_pred")
        if pred_label == "PROOF_SEARCH_DEPTH_GAP":
            tags.append("depth_gap")
        if pred_label == "MISSING_BRIDGE_LEMMA_CANDIDATE":
            tags.append("missing_bridge")
        if ns not in ("Set",):
            tags.append("non_set_namespace")
        if entropy is not None and entropy >= 0.9:
            tags.append("high_uncertainty")
        if cid and triage_of.get(cid) == "PROOF_SEARCH_DEPTH_GAP":
            tags.append("triage_depth_gap")
        if in_tr1:
            tags.append("in_tr1_training")

        row = {
            "full_name": fn,
            "file_path": fp,
            "namespace": ns,
            "source": sources[fn],
            "known_rc2_status": status,
            "in_tr1_training": in_tr1,
            "tr1_label": (ex or {}).get("label"),
            "tr1_pred_source": pred_source,
            "tr1_top_predictions": top_preds,
            "tr1_predicted_label": pred_label,
            "tr1_entropy": entropy,
            "tr1_margin": margin,
            "sf4_cluster": cid,
            "sf4_cluster_triage": triage_of.get(cid),
            "goal_features": goal_features,
            "selection_tags": tags,
        }
        # exclude SX3 production-subsumed cases entirely; exclude missing-path unknowns
        if (ex or {}).get("label") == "SX3_PRODUCTION_SUBSUMED":
            excluded.append({"full_name": fn, "reason": "sx3_production_subsumed"})
            continue
        if not fp:
            excluded.append({"full_name": fn, "reason": "missing_file_path"})
            continue
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_pool), exist_ok=True)
    with open(args.out_pool, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ----
    import collections
    by_status = collections.Counter(r["known_rc2_status"] for r in rows)
    by_ns = collections.Counter(r["namespace"] for r in rows)
    by_pred = collections.Counter(r["tr1_predicted_label"] for r in rows)
    by_cluster = collections.Counter(r["sf4_cluster"] for r in rows)
    n_in_tr1 = sum(1 for r in rows if r["in_tr1_training"])
    summary = {
        "num_candidates": len(rows),
        "num_excluded": len(excluded),
        "excluded": excluded,
        "overlap_with_tr1_training": {"count": n_in_tr1, "fraction": round(n_in_tr1 / max(1, len(rows)), 3),
                                      "note": "All candidates already labelled in TR1 -> fresh frontier exhausted; "
                                              "kept as controls / re-probe targets per spec."},
        "router_loaded": router.ok,
        "by_known_rc2_status": dict(by_status),
        "by_namespace": dict(by_ns),
        "by_predicted_label": dict(by_pred),
        "by_sf4_cluster": {k: v for k, v in by_cluster.items() if k},
        "sources": sorted({s for r in rows for s in r["source"]}),
        "inputs": {"tr1_active_learning": TR1_AL, "tr1_queue": TR1_QUEUE, "tr1_pred": TR1_PRED,
                   "tr1_examples": TR1_EX, "sf4_pool": SF4_POOL, "sf4_confirmation": SF4_CONF,
                   "sf4_clusters": SF4_CLUST, "sf4_missing_lemma": SF4_MISS,
                   "frontier_paths": FRONT_PATHS, "frontier_classified": FRONT_CLS},
    }
    json.dump(summary, open(args.out_summary_json, "w"), indent=2)

    L = ["# TR2 candidate pool", "",
         f"- **candidates:** {len(rows)}  ·  excluded: {len(excluded)}",
         f"- **overlap with TR1 training:** {n_in_tr1}/{len(rows)} "
         f"({summary['overlap_with_tr1_training']['fraction']}) — fresh frontier **exhausted**; kept as controls.",
         f"- router loaded for fresh scoring: {router.ok}", "",
         "## Known RC2 status", ""]
    for k, v in by_status.most_common():
        L.append(f"- `{k}`: {v}")
    L += ["", "## Namespace", ""]
    for k, v in by_ns.most_common():
        L.append(f"- `{k}`: {v}")
    L += ["", "## Router predicted label", ""]
    for k, v in by_pred.most_common():
        L.append(f"- `{k}`: {v}")
    L += ["", "## SF4 cluster", ""]
    for k, v in by_cluster.most_common():
        if k:
            L.append(f"- `{k}`: {v}")
    if excluded:
        L += ["", "## Excluded", ""]
        for e in excluded:
            L.append(f"- `{e['full_name']}` — {e['reason']}")
    open(args.out_summary_md, "w").write("\n".join(L))
    print(f"[tr2-pool] candidates={len(rows)} excluded={len(excluded)} in_tr1={n_in_tr1} "
          f"status={dict(by_status)} router={router.ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
