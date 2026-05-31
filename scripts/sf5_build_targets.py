#!/usr/bin/env python3
"""SF5 Part 2 — build the missing-bridge target set.

A target is included iff it is (a) a confirmed *literal* RC2 failure
(`rc2_failure_confirmation.json`, CONFIRMED_RC2_FAILURE), and (b) flagged as a
missing-bridge candidate by TR2 (MISSING_BRIDGE_LEMMA_CANDIDATE) or by an SF4
POSSIBLE_MISSING_BRIDGE_LEMMA cluster. Deduplicated by full_name.

No live Lean is run here; we reuse the verified SF4/TR2 artifacts. Goal text and
the last error are recovered from the verified probe error strings (the Lean error
echoes the goal after `⊢`).
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACED_ROOT = os.path.expanduser(
    "~/.cache/lean_dojo/leanprover-community-mathlib4-"
    "29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4")


def _p(*a):
    return os.path.join(_REPO, *a)


TR2_ATTR = "project/evolve/experiments/tr2/out/tr2_attributed_outcomes.json"
TR2_LIVE = "project/evolve/experiments/tr2/out/tr2_live_probe_results.json"
SF4_CONF = "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json"
SF4_CLUST = "project/evolve/experiments/sf4/out/rc2_failure_clusters.json"
SF4_CANDS = "project/evolve/experiments/sf4/out/sf4_missing_lemma_candidates.json"


def _load(path):
    return json.load(open(_p(path)))


def _features(text):
    t = (text or "")
    low = t.lower()
    return {
        "has_iff": ("↔" in t) or ("iff" in low),
        "has_subset": ("⊆" in t) or ("subset" in low),
        "has_ssubset": ("⊂" in t) or ("ssubset" in low),
        "has_monotone": "monotone" in low,
        "has_strictmono": ("strictmono" in low) or ("strict_mono" in low),
        "has_set": ("set" in low) or ("∈" in t) or ("∪" in t) or ("∩" in t),
        "has_singleton": ("singleton" in low) or ("{" in t),
        "has_insert": "insert" in low,
        "has_compl": ("compl" in low) or ("ᶜ" in t),
        "has_pair": "pair" in low,
        "has_ite": ("ite" in low) or ("if " in low),
        "has_union": ("union" in low) or ("∪" in t),
        "has_empty": "empty" in low,
    }


_DECL_KW = ("theorem", "lemma", "def")


_PROOF_STEP_KW = ("rw ", "rw[", "simp", "exact", "refine", "apply", "ext", "by_cases",
                  "rintro", "intro", "obtain", "constructor", "cases", "induction",
                  "rcases", "aesop", "tauto", "omega", "decide", "rfl", "field_simp")


def _decl_from_source(file_path, full_name, root=_TRACED_ROOT):
    """Recover (statement_signature, proof_body) from the traced Mathlib source.

    Returns (stmt, proof) where stmt is the signature up to `:=` and proof is the
    text after `:=` (best-effort, bounded). Either may be None."""
    if not file_path or not root or not os.path.isdir(root):
        return None, None
    fp = os.path.join(root, file_path)
    if not os.path.exists(fp):
        return None, None
    short = full_name.split(".")[-1]
    pat = re.compile(r"^\s*(?:protected\s+|@\[[^\]]*\]\s*)*(?:theorem|lemma|def)\s+"
                     + re.escape(short) + r"\b")
    try:
        lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None, None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            block = []
            for j in range(i, min(i + 22, len(lines))):
                seg = lines[j]
                if j > i and re.match(r"^\s*(?:protected\s+|@\[|theorem |lemma |def |"
                                      r"namespace |end |section |/--|#align)", seg):
                    break
                block.append(seg)
            text = "\n".join(block)
            idx = text.find(":=")
            if idx == -1:
                return re.sub(r"\s+", " ", text).strip(), None
            stmt = re.sub(r"\s+", " ", text[:idx]).strip()
            proof = text[idx + 2:].strip()
            proof = re.sub(r"#align.*$", "", proof, flags=re.S).strip()
            return stmt, proof
    return None, None


def _proof_analysis(proof):
    """Estimate proof step count + first tactic from a `by ...` proof body."""
    if not proof:
        return {"has_source_proof": False, "num_steps": 0, "first_tactic": None,
                "is_term_proof": False, "snippet": None}
    body = proof.strip()
    is_term = not body.startswith("by")
    inner = body[2:].strip() if body.startswith("by") else body
    # split into rough tactic steps on newlines / ';' / '<;>' / bullet markers
    raw = re.split(r"[\n;]|<;>|·", inner)
    steps = [s.strip() for s in raw if s.strip()]
    first = None
    for s in steps:
        for kw in _PROOF_STEP_KW:
            if s.startswith(kw.strip()):
                first = s[:40]
                break
        if first:
            break
    return {
        "has_source_proof": True,
        "num_steps": len(steps),
        "first_tactic": first or (steps[0][:40] if steps else None),
        "is_term_proof": is_term,
        "snippet": re.sub(r"\s+", " ", body)[:200],
    }


_GOAL_RE = re.compile(r"⊢[^\n]*")


def _extract_goal(errors):
    """Best goal text from verified probe error strings (Lean echoes `⊢ ...`)."""
    best = None
    for e in errors:
        if not e:
            continue
        for m in _GOAL_RE.findall(e):
            g = m.strip()
            if best is None or len(g) > len(best):
                best = g
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-targets", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    tr2 = _load(TR2_ATTR)
    tr2_live = _load(TR2_LIVE)
    conf = _load(SF4_CONF)
    clusters = _load(SF4_CLUST)
    sf4_cands = _load(SF4_CANDS)

    # confirmed literal-RC2 failures only
    confirmed = {
        r["full_name"]: r
        for r in conf["results"]
        if r.get("classification") == "CONFIRMED_RC2_FAILURE"
    }

    # candidate names: TR2 missing-bridge + SF4 possible-missing-bridge cluster members
    tr2_mb = {
        r["full_name"]
        for r in tr2["records"]
        if r.get("classification") == "MISSING_BRIDGE_LEMMA_CANDIDATE"
    }
    sf4_mb = set()
    for tri in sf4_cands.get("triage", []):
        if tri.get("category") == "POSSIBLE_MISSING_BRIDGE_LEMMA":
            sf4_mb.update(tri.get("members", []))

    candidate_names = (tr2_mb | sf4_mb) & set(confirmed)

    # cluster lookup
    name2cluster = {}
    for cl in clusters["clusters"]:
        for m in cl["members"]:
            name2cluster[m] = cl

    # tr2 live errors per theorem (for goal/error recovery)
    live_by = {r["full_name"]: r for r in tr2_live["results"]}
    conf_by = confirmed

    targets = []
    for fn in sorted(candidate_names):
        cr = conf_by[fn]
        lr = live_by.get(fn, {})
        # gather all error strings from verified probes/controls for this theorem
        errs = []
        for bucket in ("controls", "depth1_subcontrols", "probes_tried"):
            for c in lr.get(bucket, []) or []:
                if c.get("error"):
                    errs.append(c["error"])
        if cr.get("error_message"):
            errs.append(cr["error_message"])
        goal = _extract_goal(errs)
        stmt, proof = _decl_from_source(cr.get("file_path"), fn)
        proof_info = _proof_analysis(proof)
        # goal_text prefers the live error-echoed goal; falls back to source statement
        goal_text = goal or stmt
        cl = name2cluster.get(fn, {})
        feat = _features(goal_text or fn)
        targets.append({
            "full_name": fn,
            "file_path": cr.get("file_path"),
            "namespace": cr.get("namespace"),
            "cluster_id": cl.get("cluster_id"),
            "cluster_label": cl.get("goal_shape"),
            "goal_text": goal_text,
            "statement_text": stmt,
            "source_proof": proof_info,
            "last_goal": goal,
            "last_error": cr.get("error_message"),
            "known_rc2_status": "failed",
            "tr2_label": "MISSING_BRIDGE_LEMMA_CANDIDATE",
            "in_tr2_missing_bridge": fn in tr2_mb,
            "in_sf4_missing_bridge_cluster": fn in sf4_mb,
            "rc2_tactics_used": cr.get("tactics_used"),
            "features": {k: feat[k] for k in (
                "has_iff", "has_subset", "has_monotone", "has_strictmono", "has_set")},
            "features_extended": feat,
        })

    # cluster rollup for manifest
    by_cluster = {}
    for t in targets:
        by_cluster.setdefault(t["cluster_id"], []).append(t["full_name"])

    manifest = {
        "generated_by": "scripts/sf5_build_targets.py",
        "rc2_wrapper": conf.get("rc2_wrapper"),
        "sources": {
            "tr2_attributed_outcomes": TR2_ATTR,
            "tr2_live_probe_results": TR2_LIVE,
            "sf4_rc2_failure_confirmation": SF4_CONF,
            "sf4_rc2_failure_clusters": SF4_CLUST,
            "sf4_missing_lemma_candidates": SF4_CANDS,
        },
        "num_confirmed_rc2_failures": len(confirmed),
        "num_tr2_missing_bridge": len(tr2_mb),
        "num_sf4_missing_bridge_cluster": len(sf4_mb),
        "num_targets": len(targets),
        "clusters": {
            cid: {"size": len(names), "members": sorted(names)}
            for cid, names in sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
        },
    }

    os.makedirs(os.path.dirname(_p(args.out_targets)), exist_ok=True)
    os.makedirs(os.path.dirname(_p(args.out_summary)), exist_ok=True)
    json.dump(targets, open(_p(args.out_targets), "w"), ensure_ascii=False, indent=2)
    json.dump(manifest, open(_p(args.out_manifest), "w"), ensure_ascii=False, indent=2)

    lines = [
        "# SF5 — missing-bridge target set",
        "",
        f"- targets: **{len(targets)}**",
        f"- confirmed literal-RC2 failures (pool): {len(confirmed)}",
        f"- TR2 MISSING_BRIDGE_LEMMA_CANDIDATE: {len(tr2_mb)}",
        f"- SF4 POSSIBLE_MISSING_BRIDGE_LEMMA cluster members: {len(sf4_mb)}",
        "",
        "## Clusters",
        "",
    ]
    for cid, info in manifest["clusters"].items():
        lines.append(f"### `{cid}` (size {info['size']})")
        for m in info["members"]:
            lines.append(f"- {m}")
        lines.append("")
    lines.append("## Targets")
    lines.append("")
    lines.append("| full_name | namespace | cluster | iff | subset | mono | goal |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in targets:
        f = t["features"]
        g = (t["goal_text"] or "")[:60].replace("|", "\\|")
        lines.append(
            f"| {t['full_name']} | {t['namespace']} | {t['cluster_id']} | "
            f"{'Y' if f['has_iff'] else ''} | {'Y' if f['has_subset'] else ''} | "
            f"{'Y' if f['has_monotone'] else ''} | {g} |")
    open(_p(args.out_summary), "w").write("\n".join(lines) + "\n")

    print(f"[sf5-targets] wrote {len(targets)} targets")
    print(f"  clusters: " + ", ".join(
        f"{cid}={info['size']}" for cid, info in manifest["clusters"].items()))


if __name__ == "__main__":
    main()
