#!/usr/bin/env python3
"""TR5 shared scoring helper.

Featurizes a (theorem, program) row IDENTICALLY to scripts/tr4_featurize_programs.py
and scores it with the saved TR4 HGB ranker (full-data model) + the saved heuristic
ranker. Used by the ranked-program-plan builder so the live search runs the same scores
TR4 reported offline. No live Lean here.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import scipy.sparse as sp
import joblib

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vectorizer block order MUST match tr4_featurize_programs.py
_BLOCK_ORDER = ["name_char", "name_tok", "goal_word", "tactic_tok", "lemma_tok", "dict"]


def _p(*a):
    return os.path.join(_REPO, *a)


def _name_text(fn):
    return " ".join(re.split(r"[._]", fn or ""))


def _lemma_text(lemmas):
    return " ".join(" ".join(re.split(r"[._]", L)) for L in (lemmas or []))


def _ns(fn):
    return fn.split(".")[0] if "." in fn else ""


def _flags(text):
    t = text or ""
    low = t.lower()
    return {
        "has_set": ("set" in low) or ("∈" in t) or ("∪" in t) or ("∩" in t) or ("⊆" in t),
        "has_finset": "finset" in low,
        "has_list": "list" in low,
        "has_nat": ("nat" in low) or ("ℕ" in t),
        "has_iff": ("↔" in t) or ("iff" in low),
        "has_subset": ("⊆" in t) or ("⊂" in t) or ("subset" in low),
        "has_disjoint": "disjoint" in low,
        "has_compl": ("compl" in low) or ("ᶜ" in t),
        "has_singleton": ("singleton" in low) or ("{" in t),
        "has_card": "card" in low,
        "has_tofinset": "tofinset" in low,
        "has_monotone": "monotone" in low,
    }


def _rank_bucket(rank):
    if rank is None:
        return "none"
    if rank == 0:
        return "r0"
    if rank <= 2:
        return "r1-2"
    if rank <= 5:
        return "r3-5"
    if rank <= 10:
        return "r6-10"
    return "r11+"


def build_row(full_name, goal_text, namespace, tactic, lemmas, family, depth,
              retrieval_rank=None, retrieval_score=None, lemma_source=None,
              source="tr5"):
    """Reconstruct the TR4-schema row dict (with the same `features`)."""
    lemmas = lemmas or []
    text_for_flags = (goal_text or "") + " " + full_name
    fl = _flags(text_for_flags)
    tl = (tactic or "").lower()
    lemma_ns_match = any(_ns(L) == namespace for L in lemmas if "." in L)
    uses_retrieved = bool(lemmas) and retrieval_rank is not None
    return {
        "full_name": full_name, "namespace": namespace, "goal_text": goal_text,
        "tactic": tactic, "used_lemmas": lemmas, "program_family": family,
        "program_depth": depth, "retrieval_rank": retrieval_rank,
        "retrieval_score": retrieval_score, "lemma_source": lemma_source,
        "source": source,
        "features": {
            **{k: bool(v) for k, v in fl.items()},
            "lemma_namespace_matches": bool(lemma_ns_match),
            "program_uses_retrieved_lemma": bool(uses_retrieved),
            "is_def_unfold": family == "def_unfold_simp",
            "is_depth2_aesop": family == "d2_simp_aesop",
            "is_d1_simp_lemma": family == "d1_simp_lemma",
            "uses_simp": "simp" in tl, "uses_rw": "rw " in tl or tl.startswith("rw"),
            "uses_exact": "exact" in tl, "uses_simpa": "simpa" in tl,
            "uses_aesop": "aesop" in tl, "uses_ext": tl.startswith("ext"),
            "uses_constructor": "constructor" in tl, "uses_intro": "intro" in tl,
            "uses_omega": "omega" in tl, "uses_nlinarith": "nlinarith" in tl,
        },
    }


def _dict_feats(r):
    f = dict(r.get("features", {}))
    d = {f"flag={k}": (1.0 if v else 0.0) for k, v in f.items()}
    d[f"ns={r.get('namespace')}"] = 1.0
    d[f"fam={r.get('program_family')}"] = 1.0
    d[f"depth={r.get('program_depth')}"] = 1.0
    d[f"lemma_src={r.get('lemma_source')}"] = 1.0
    d[f"source={r.get('source')}"] = 1.0
    rb = _rank_bucket(r.get("retrieval_rank"))
    d[f"rankbucket={rb}"] = 1.0
    sc = r.get("retrieval_score")
    d["retrieval_score"] = float(sc) if isinstance(sc, (int, float)) else 0.0
    d["retrieval_rank"] = float(r["retrieval_rank"]) if r.get("retrieval_rank") is not None else 99.0
    d["has_retrieval"] = 1.0 if r.get("retrieval_rank") is not None else 0.0
    fam = r.get("program_family")
    d[f"int_rankbucket×fam={rb}|{fam}"] = 1.0
    nsmatch = r.get("features", {}).get("lemma_namespace_matches")
    d[f"int_nsmatch×fam={int(bool(nsmatch))}|{fam}"] = 1.0
    d[f"int_usesretr×fam={int(bool(r.get('features',{}).get('program_uses_retrieved_lemma')))}|{fam}"] = 1.0
    return d


class RankerScorer:
    def __init__(self, vectorizers_path, model_path, heuristic_path=None):
        self.vec = joblib.load(_p(vectorizers_path) if not os.path.isabs(vectorizers_path) else vectorizers_path)
        self.model = joblib.load(_p(model_path) if not os.path.isabs(model_path) else model_path)
        self.heuristic = None
        if heuristic_path:
            hp = _p(heuristic_path) if not os.path.isabs(heuristic_path) else heuristic_path
            if os.path.exists(hp):
                self.heuristic = json.load(open(hp))

    def _featurize(self, rows):
        nc = self.vec["name_char"].transform([_name_text(r["full_name"]) for r in rows])
        nt = self.vec["name_tok"].transform([_name_text(r["full_name"]) for r in rows])
        gw = self.vec["goal_word"].transform([(r.get("goal_text") or "") for r in rows])
        tt = self.vec["tactic_tok"].transform([(r.get("tactic") or "") for r in rows])
        lt = self.vec["lemma_tok"].transform([_lemma_text(r.get("used_lemmas")) for r in rows])
        dd = self.vec["dict"].transform([_dict_feats(r) for r in rows])
        blocks = {"name_char": nc, "name_tok": nt, "goal_word": gw,
                  "tactic_tok": tt, "lemma_tok": lt, "dict": dd}
        return sp.hstack([blocks[k] for k in _BLOCK_ORDER]).tocsr()

    def score(self, rows):
        """Return HGB P(success) for each row."""
        if not rows:
            return np.array([])
        X = self._featurize(rows)
        Xd = X.toarray() if hasattr(self.model, "_predict_from_X") or True else X
        try:
            return self.model.predict_proba(Xd)[:, 1]
        except Exception:
            return self.model.predict_proba(X)[:, 1]

    def heuristic_score(self, r):
        """Replicate the TR4 rule heuristic exactly (see heuristic_ranker.json rules):
        +win_family, +0.4 ns_match, +0.5 uses_retrieved, +max(0,0.6-0.05*rank),
        -0.3 if no retrieval, -0.1*(depth-1), +0.3 def_unfold."""
        h = self.heuristic or {}
        fam_w = (h.get("family_weights") or {})
        feats = r.get("features", {})
        s = 0.0
        s += fam_w.get(r.get("program_family"), 0.0)
        if feats.get("lemma_namespace_matches"):
            s += 0.4
        rank = r.get("retrieval_rank")
        if feats.get("program_uses_retrieved_lemma") and rank is not None:
            s += 0.5
            s += max(0.0, 0.6 - 0.05 * float(rank))
        else:
            s -= 0.3
        s -= 0.1 * (float(r.get("program_depth", 1)) - 1.0)
        if r.get("program_family") == "def_unfold_simp":
            s += 0.3
        return round(float(s), 4)
