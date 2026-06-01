#!/usr/bin/env python3
"""SF1 stage (c): real deterministic frontier classifier.

Reads the open-frontier JSONL produced by Stage A/B and tags every declaration
with namespace tags, syntactic / proof-shape tags, and deterministic candidate
family scores. No LeanDojo live tracing is required: when `statement`/`type` are
absent (as in artifact-mode catalogs), classification falls back to the
declaration name + namespace and records a low confidence (<= 0.5).

Outputs:
  classified_frontier.jsonl   one record per declaration (see schema below)
  classification_summary.json  aggregate histograms

Determinism: pure function of the input records; no RNG, no clock.

SAFETY: never reads/writes any production config (RC1 wrapper, NS9 genome,
NS24 router, REL1 reports). Writes only under the SF1 out dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sf1_common as C
    _read = C.read_json_or_jsonl
    _write = C.write_jsonl
    _ensure = C.ensure_parent_dir
    _extract = C.extract_decl_names_from_record
except Exception:  # pragma: no cover - defensive fallback
    def _read(path):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        rows.append(line)
        return rows

    def _ensure(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        return path

    def _write(rows, path):
        _ensure(path)
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

    def _extract(rec, key_hints=None):
        if isinstance(rec, str):
            return [rec]
        if isinstance(rec, dict):
            for k in ("decl_name", "full_name", "theorem_name", "name", "theorem"):
                v = rec.get(k)
                if isinstance(v, str) and v:
                    return [v]
        return []

SEED = 1729

# Candidate families we score (registry ids + the discovery placeholder).
FAMILIES = [
    "ns9_base_wrapper",
    "rc1_production_stack",
    "wx3_multiset_induction",
    "mx2_set_finite_tofinset_aesop",
    "ax4_learned_symbolic_selector_off_by_default",
    "sx1_depth2_sequence_search_off_by_default",
    "broad_set_aesop_rejected",
    "mx1_set_finset_symbolic_rejected",
    "future_failure_driven_lemma_candidate",
]
REJECTED_OR_OFF = {
    "broad_set_aesop_rejected",
    "mx1_set_finset_symbolic_rejected",
    "ax4_learned_symbolic_selector_off_by_default",
    "sx1_depth2_sequence_search_off_by_default",
}

# namespace bucket -> namespace tag
NAMESPACE_TAGS = {
    "Set": "has_set", "Finset": "has_finset", "Multiset": "has_multiset",
    "List": "has_list", "Option": "has_option", "Nat": "has_nat",
    "Int": "has_int",
}
ORDER_TOKENS = {"order", "le", "lt", "covby", "wcovby", "min", "max", "sup",
                "inf", "lattice", "monotone", "strictmono"}
ALGEBRA_TOKENS = {"add", "mul", "sub", "neg", "pow", "mod", "div", "gcd", "lcm",
                  "ring", "group", "monoid", "field", "smul", "nsmul", "zsmul",
                  "dvd", "units", "cast"}
TOPOLOGY_TOKENS = {"topology", "continuous", "open", "closed", "nhds", "compact",
                   "tendsto", "filter"}
LOGIC_TOKENS = {"iff", "and", "or", "not", "forall", "exists", "imp", "decidable",
                "true", "false", "ne", "eq"}
FUNCTION_TOKENS = {"comp", "injective", "surjective", "bijective", "function",
                   "leftinverse", "rightinverse", "involutive", "id", "map"}


def _name_of(rec):
    cand = _extract(rec)
    if cand:
        return cand[0]
    if isinstance(rec, dict):
        for k in ("decl_name", "full_name", "name", "theorem"):
            v = rec.get(k)
            if isinstance(v, str) and v:
                return v
    return None


def _namespace_of(rec, name):
    if isinstance(rec, dict):
        ns = rec.get("namespace") or rec.get("namespace_bucket")
        if isinstance(ns, str) and ns:
            return ns
    if name and "." in name:
        return name.split(".")[0]
    return "GENERAL_FRONTIER"


def _statement_of(rec):
    if isinstance(rec, dict):
        for k in ("statement", "type", "theorem_type", "stmt", "goal", "state_pp"):
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _tokens(name):
    toks = set()
    if not name:
        return toks
    for part in name.replace(".", "_").split("_"):
        p = part.strip().lower()
        if p:
            toks.add(p)
    return toks


def classify(rec):
    name = _name_of(rec)
    namespace = _namespace_of(rec, name)
    statement = _statement_of(rec)
    toks = _tokens(name)
    s = (statement or "").lower()

    basis = ["name"]
    if isinstance(rec, dict) and rec.get("namespace"):
        basis.append("namespace")
    if statement is not None:
        basis.append("type" if (isinstance(rec, dict) and rec.get("type")) else "statement")

    tags = set()

    # ---- namespace tags ----
    nstag = NAMESPACE_TAGS.get(namespace.split(".")[0]) if namespace else None
    if nstag:
        tags.add(nstag)
    low_ns = (namespace or "").lower()
    if "set" in low_ns and namespace not in ("Finset", "Multiset"):
        tags.add("has_set")
    if toks & ORDER_TOKENS:
        tags.add("has_order")
    if toks & ALGEBRA_TOKENS:
        tags.add("has_algebra")
    if toks & TOPOLOGY_TOKENS or "topology" in low_ns:
        tags.add("has_topology")
    if toks & LOGIC_TOKENS:
        tags.add("has_logic")
    if toks & FUNCTION_TOKENS:
        tags.add("has_function")

    def has(*kw):
        return any(k in toks for k in kw) or any(k in s for k in kw)

    # ---- syntactic / proof-shape tags (name- and statement-driven) ----
    if "∀" in s or has("forall"):
        tags.add("has_forall")
    if "∃" in s or has("exists"):
        tags.add("has_exists")
    if "↔" in s or has("iff"):
        tags.add("has_iff")
    if "=" in s or has("eq"):
        tags.add("has_eq")
    if "≤" in s or has("le"):
        tags.add("has_le")
    if "<" in s or has("lt"):
        tags.add("has_lt")
    if "∧" in s or has("and"):
        tags.add("has_and")
    if "∨" in s or has("or"):
        tags.add("has_or")
    if "¬" in s or has("not", "ne"):
        tags.add("has_not")
    if "∈" in s or has("mem", "membership"):
        tags.add("has_membership")
    if "⊆" in s or has("subset"):
        tags.add("has_subset")
    if "∪" in s or has("union"):
        tags.add("has_union")
    if "∩" in s or has("inter"):
        tags.add("has_inter")
    if has("insert"):
        tags.add("has_insert")
    if has("empty", "nil", "zero"):
        tags.add("has_empty")
    if has("singleton"):
        tags.add("has_singleton")
    if has("card", "length", "count"):
        tags.add("has_cardinality")
    if has("coe", "cast"):
        tags.add("has_coe")
    if has("tofinset"):
        tags.add("has_toFinset")
    if has("finite"):
        tags.add("has_finite")

    is_multiset = "has_multiset" in tags
    is_set = "has_set" in tags
    is_finset = "has_finset" in tags
    is_arith = bool(tags & {"has_algebra"}) or namespace in ("Nat", "Int")

    # ---- likely-tactic heuristics ----
    tags.add("likely_simp")  # simp is the universal first probe
    if is_set or is_finset or ("has_logic" in tags) or ("has_membership" in tags) \
            or ("has_subset" in tags):
        tags.add("likely_aesop")
    if is_arith or ("has_le" in tags) or ("has_lt" in tags) or has("mod", "dvd"):
        if namespace in ("Nat", "Int") or is_arith:
            tags.add("likely_omega")
    if is_finset or namespace in ("Option", "Bool") or has("ite", "dite", "cases"):
        tags.add("likely_cases")
    if is_multiset or namespace in ("List", "Nat") or has("induction", "rec",
                                                          "cons", "foldr", "foldl"):
        tags.add("likely_induction")
    if has("ext") or (is_set and "has_eq" in tags) or (is_finset and "has_eq" in tags):
        tags.add("likely_extensionality")
    if is_set and ("has_finite" in tags or "has_toFinset" in tags):
        tags.add("likely_set_finite_bridge")
    if is_multiset and (has("induction", "rec", "cons") or "likely_induction" in tags):
        tags.add("likely_multiset_induction")

    # ---- candidate family scores (deterministic, 0..1) ----
    sc = {f: 0.0 for f in FAMILIES}

    # rc1 is the control; it covers everything the base policy covers.
    sc["rc1_production_stack"] = 0.40
    sc["ns9_base_wrapper"] = 0.35
    if is_arith or namespace in ("Nat", "Int") or "has_order" in tags:
        sc["rc1_production_stack"] = max(sc["rc1_production_stack"], 0.70)
        sc["ns9_base_wrapper"] = max(sc["ns9_base_wrapper"], 0.65)

    # wx3 Multiset induction
    if is_multiset:
        sc["wx3_multiset_induction"] = 0.55
        if "likely_multiset_induction" in tags:
            sc["wx3_multiset_induction"] = 0.90
        elif has("cons", "induction", "rec", "bind", "map", "foldr", "foldl"):
            sc["wx3_multiset_induction"] = 0.75
        sc["rc1_production_stack"] = max(sc["rc1_production_stack"], 0.50)
        # ax4 learned selector shadows the same surface (off by default)
        sc["ax4_learned_symbolic_selector_off_by_default"] = \
            min(0.6, sc["wx3_multiset_induction"] - 0.2)

    # mx2 narrow Set.Finite / toFinset aesop
    if is_set:
        if "likely_set_finite_bridge" in tags or "has_toFinset" in tags or "has_finite" in tags:
            sc["mx2_set_finite_tofinset_aesop"] = 0.85
        elif tags & {"has_membership", "has_subset", "has_union", "has_inter"}:
            sc["mx2_set_finite_tofinset_aesop"] = 0.45
        sc["rc1_production_stack"] = max(sc["rc1_production_stack"], 0.45)
        # rejected broad-Set / Set-Finset symbolic families: score the surface
        # they *target* but they are rejected/off; never promote.
        if "likely_extensionality" in tags or "has_eq" in tags:
            sc["broad_set_aesop_rejected"] = 0.50
            sc["mx1_set_finset_symbolic_rejected"] = 0.45

    if is_finset:
        sc["rc1_production_stack"] = max(sc["rc1_production_stack"], 0.45)
        if "likely_extensionality" in tags:
            sc["mx1_set_finset_symbolic_rejected"] = 0.50

    # ---- confidence ----
    if statement is not None:
        confidence = 0.80 if len(tags) >= 4 else 0.65
    else:
        # name + namespace only
        confidence = 0.50 if (nstag and len(tags) >= 5) else 0.35

    # failure-driven lemma candidate: we have RC1 relevance but NO strong
    # productive family match -> the surface where a missing lemma likely lives.
    productive = max(sc["wx3_multiset_induction"],
                     sc["mx2_set_finite_tofinset_aesop"])
    rc1_rel = sc["rc1_production_stack"]
    if productive < 0.5 and rc1_rel >= 0.4:
        sc["future_failure_driven_lemma_candidate"] = round(
            0.4 + (0.5 - productive) + (0.5 - confidence) * 0.5, 3)
        sc["future_failure_driven_lemma_candidate"] = min(
            0.85, sc["future_failure_driven_lemma_candidate"])

    top = max(sc, key=lambda k: (sc[k], k))
    notes = []
    if statement is None:
        notes.append("no statement/type available; classified from name+namespace (low confidence)")
    if top in REJECTED_OR_OFF:
        notes.append(f"top family '{top}' is rejected/off-by-default; not promotion-eligible")
    if sc["future_failure_driven_lemma_candidate"] >= 0.5:
        notes.append("flagged as failure-driven lemma-discovery candidate (weak productive-family match)")

    return {
        "decl_name": name,
        "namespace": namespace,
        "statement": statement,
        "tags": sorted(tags),
        "candidate_family_scores": {k: round(v, 3) for k, v in sc.items()},
        "top_candidate_family": top,
        "classification_confidence": round(confidence, 3),
        "classification_basis": basis,
        "notes": notes,
    }


def summarize(records):
    from collections import Counter
    ns = Counter(r["namespace"] for r in records)
    tag_counts = Counter()
    for r in records:
        tag_counts.update(r["tags"])
    top_fam = Counter(r["top_candidate_family"] for r in records)
    fam_hist = Counter()
    for r in records:
        for f, v in r["candidate_family_scores"].items():
            if v >= 0.5:
                fam_hist[f] += 1
    low_conf = sum(1 for r in records if r["classification_confidence"] <= 0.5)
    set_ct = sum(1 for r in records if "has_set" in r["tags"])
    ms_ct = sum(1 for r in records if "has_multiset" in r["tags"])
    return {
        "seed": SEED,
        "total_classified": len(records),
        "counts_by_namespace": dict(ns.most_common()),
        "counts_by_tag": dict(tag_counts.most_common()),
        "top_candidate_family_counts": dict(top_fam.most_common()),
        "candidate_family_histogram_ge_0_5": dict(fam_hist.most_common()),
        "low_confidence_count": low_conf,
        "set_frontier_count": set_ct,
        "multiset_frontier_count": ms_ct,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (c): real deterministic frontier classifier.")
    p.add_argument("--input", "--frontier",
                   default="project/evolve/experiments/sf1/out/real/frontier.jsonl",
                   help="Input frontier JSONL from Stage A/B.")
    p.add_argument("--out",
                   default="project/evolve/experiments/sf1/out/real/classified_frontier.jsonl")
    p.add_argument("--summary-out",
                   default="project/evolve/experiments/sf1/out/real/classification_summary.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"[sf1:classify] ERROR: frontier not found: {args.input}", file=sys.stderr)
        return 2
    rows = _read(args.input)
    records = [classify(r) for r in rows if _name_of(r)]
    _write(records, args.out)
    summary = summarize(records)
    _ensure(args.summary_out)
    with open(args.summary_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[sf1:classify] seed={SEED} classified {len(records)} declarations "
          f"(set={summary['set_frontier_count']} multiset={summary['multiset_frontier_count']} "
          f"low_conf={summary['low_confidence_count']}) -> {args.out}")
    print(f"[sf1:classify] top-family counts: {summary['top_candidate_family_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
