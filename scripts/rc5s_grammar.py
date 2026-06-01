#!/usr/bin/env python3
"""RC5S shared strict-grammar classifier (imported by filter / generate / runner).

The RC5H dynamic stage leaked a broad TR6 grammar (simp_all combos, depth-3 try chains, bare
aesop/omega/tauto) that stalled LeanDojo. RC5S enforces a strict low-risk grammar: every program
must match one of the allowed patterns exactly, with a real lemma where required, no simp_all, no
depth-3 try chains, and `<;> aesop` only on historically-safe namespaces.

classify_program(tactic, namespace, policy) -> (klass, allowed_bool) with klass in:
  POLICY_ALLOWED / REMOVED_STALL_RISK / REMOVED_OFF_POLICY / REMOVED_NAMESPACE_DISABLED.
The caller adds REMOVED_LOW_CONFIDENCE / REMOVED_DUPLICATE / NEEDS_REVIEW.
"""
from __future__ import annotations

import re

# allowed low-risk patterns (anchored). `<L>` = a `[...]` lemma list or a bare lemma name.
_LEMMA = r"[^\]]+"
ALLOWED_PATTERNS = [
    ("exact_L",            re.compile(r"^exact\s+\S.*$")),
    ("simpa_using_L",      re.compile(r"^simpa\s+using\s+\S.*$")),
    ("simpa_L",            re.compile(r"^simpa\s+\[" + _LEMMA + r"\]$")),
    ("simp_L",             re.compile(r"^simp\s+\[" + _LEMMA + r"\]$")),
    ("rw_L",               re.compile(r"^rw\s+\[" + _LEMMA + r"\]$")),
    ("simp_L_aesop",       re.compile(r"^simp\s+\[" + _LEMMA + r"\]\s+<;>\s+aesop$")),
    ("rw_L_aesop",         re.compile(r"^rw\s+\[" + _LEMMA + r"\]\s+<;>\s+aesop$")),
    ("ext_simp_L",         re.compile(r"^ext\s+\w+\s+<;>\s+simp\s+\[" + _LEMMA + r"\]$")),
    ("constructor_aesop",  re.compile(r"^constructor\s+<;>\s+intro\s+\w+\s+<;>\s+aesop$")),
]
# patterns whose `<;> aesop` tail is namespace-gated (historically-safe only)
_AESOP_PATTERNS = {"simp_L_aesop", "rw_L_aesop", "constructor_aesop"}
DEFAULT_AESOP_NAMESPACES = ("Set", "Finset", "List", "Multiset")
DEFAULT_ALLOWED_NAMESPACES = ("Set", "Finset", "List", "Multiset", "Nat")

_STALL_MARKERS = ("simp_all", "<;> try", " try ", "tauto", "decide", "norm_num <;>")


def _pattern_of(tactic):
    t = (tactic or "").strip()
    for pid, rx in ALLOWED_PATTERNS:
        if rx.match(t):
            return pid
    return None


def classify_program(tactic, namespace, policy=None):
    """Return (klass, allowed). policy may carry aesop_namespaces / allowed_namespaces."""
    t = (tactic or "").strip()
    pol = policy or {}
    allowed_ns = set(pol.get("allowed_namespaces", DEFAULT_ALLOWED_NAMESPACES))
    aesop_ns = set(pol.get("aesop_namespaces", DEFAULT_AESOP_NAMESPACES))

    # 1) stall-risk markers (highest priority — these are the blockers RC5H hit)
    low = t.lower()
    if any(m in low for m in ("simp_all",)):
        return "REMOVED_STALL_RISK", False
    if "<;> try" in low or low.startswith("try ") or " try " in low:
        return "REMOVED_STALL_RISK", False
    # depth-3 chains (>=2 `<;>`) that are not the single allowed constructor pattern
    if t.count("<;>") >= 2 and _pattern_of(t) != "constructor_aesop":
        return "REMOVED_STALL_RISK", False

    pid = _pattern_of(t)
    if pid is None:
        return "REMOVED_OFF_POLICY", False  # bare aesop/omega/nlinarith/tauto, odd combos

    # namespace gate
    ns = (namespace or "").split(".")[0]
    if ns and ns not in allowed_ns:
        return "REMOVED_NAMESPACE_DISABLED", False
    # aesop-tail namespace gate
    if pid in _AESOP_PATTERNS and ns and ns not in aesop_ns:
        return "REMOVED_NAMESPACE_DISABLED", False
    return "POLICY_ALLOWED", True


def pattern_of(tactic):
    return _pattern_of(tactic)
