"""Template verifier — best-effort filter for tactic templates that
reference constants known to be unavailable or known to produce type
mismatches in the eval environment.

This is the template-side analog of `premise_retriever._UNAVAILABLE_LEMMAS`:
where that denylist gates *retrieved* premises before they are wrapped into
tactic forms, this module gates *hard-coded* template strings (the
`theorem_family_tactics` and `fallback_tactics` shipped on the candidate).

It is NOT a Lean type checker. It does two cheap things:

  1. Extract literal constants from a template (using the same regex
     machinery as `premise_retriever.extract_premises_from_tactic` plus a
     special case for `rw [Foo arg, Bar]` where `arg` is a hypothesis).
  2. Compare those constants against two static sets:
       - KNOWN_UNAVAILABLE_CONSTANTS: produce `unknown constant '...'` in
         Mathlib/Data/Nat/Defs.lean's import closure at one of the v4.5
         eval positions. Populated from observed errors.
       - KNOWN_TYPE_MISMATCH_CONSTANTS: available but the way our
         templates use them is type-incorrect across every observed
         instance. Populated from observed `type mismatch` errors.

A template fails verification if it references any constant in either set
*outside* a comment / hypothesis position. Failures are returned with the
specific offending constant so the caller can log a diagnostic.

The verifier never raises. It is allowed to be conservative — false
positives only cost a few candidate tactics, and the original v4.5 default
configuration is always recoverable by passing verification_enabled=False.

Caller pattern:

    from evolve.template_verifier import filter_templates, default_unavailable

    kept, dropped = filter_templates(
        templates=family_div_templates,
        unavailable=default_unavailable(),
    )
"""

from __future__ import annotations

import re
from typing import Iterable

# Reuse the premise_retriever extractor for the bracketed forms; add a
# light apply/exact extractor that captures the *first* identifier after
# the keyword (the lemma name, before any args).
from premise_retriever import extract_premises_from_tactic, _UNAVAILABLE_LEMMAS


# Constants observed to error with `unknown constant '...'` in the v4.5
# eval traces, on at least one nat_defs_medium target. Combined with the
# premise-retrieval denylist so both layers stay in sync.
_OBSERVED_UNKNOWN_CONSTANTS: set[str] = {
    "Nat.left_comm",
    "Nat.div_le_div_right",  # already in _UNAVAILABLE_LEMMAS, kept for clarity
    "Nat.div_le_iff_le_mul",
    "Nat.div_eq_zero_iff",
    "Nat.div_lt_one_iff",
    "Nat.div_pos",
    "Nat.div_pos_iff",
    "Nat.dvd_iff_div_mul_eq",
}


# Constants observed to produce `type mismatch` errors across every
# attempt in the v4.5 traces — the constant exists, but the *form* used
# in our templates passes the wrong types. Listing them as
# type_mismatch (rather than unavailable) signals to the diagnostic that
# the right fix is to rewrite the template rather than wait for the
# constant to become reachable.
_OBSERVED_TYPE_MISMATCH_CONSTANTS: set[str] = {
    "Nat.le_refl",          # induction step constructor used as :  Nat.le_refl _
    "Nat.div_le_succ_div",  # `ih.trans (Nat.div_le_succ_div _ _)` — wrong arity
}


def default_unavailable() -> set[str]:
    """Union of the premise-retrieval unavailable set and the
    template-observed unknown-constant set."""
    return set(_UNAVAILABLE_LEMMAS) | set(_OBSERVED_UNKNOWN_CONSTANTS)


def default_type_mismatch() -> set[str]:
    return set(_OBSERVED_TYPE_MISMATCH_CONSTANTS)


# A template can contain placeholders like `{var}`, `{hyp_le}`,
# `{hyp_pos}`, `{hyp_ne_zero}` — these are substituted at render time and
# must NOT be flagged as constants. Strip them before extraction.
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][\w_]*\}")


def extract_template_constants(template: str) -> set[str]:
    """Extract the set of qualified lemma/constant names referenced
    inside a template's `simp/rw/apply/exact` arguments.

    Placeholders (`{var}`, `{hyp_le}`, ...) are stripped before extraction
    so they don't leak into the result. Identifiers that look like local
    hypotheses (`h`, `hba`, `h_step`) are skipped by the underlying
    extractor.

    Returns a *set* — duplicate references inside a single template don't
    inflate the diagnostic.
    """
    stripped = _PLACEHOLDER_RE.sub("__PLACEHOLDER__", template or "")
    out: set[str] = set()
    for c in extract_premises_from_tactic(stripped):
        # `rw [Foo {hyp_pos}, Bar]` after placeholder substitution becomes
        # `Foo __PLACEHOLDER__` — _BRACKET_ARGS sees that as one arg.
        # Keep only the head identifier (the lemma name proper).
        head = c.split()[0].strip() if c else ""
        if not head or head == "__PLACEHOLDER__":
            continue
        out.add(head)
    return out


def verify_template(
    template: str,
    unavailable: set[str] | None = None,
    type_mismatch: set[str] | None = None,
) -> tuple[bool, dict]:
    """Return (kept, diagnostic).

    `kept` is True iff the template references no constant in either set.
    `diagnostic` always contains the keys
      `template`, `constants`, `unavailable_hits`, `type_mismatch_hits`,
      `reason`. `reason` is "" when kept and a short label otherwise.
    """
    unavailable = unavailable if unavailable is not None else default_unavailable()
    type_mismatch = type_mismatch if type_mismatch is not None else default_type_mismatch()
    constants = sorted(extract_template_constants(template))
    bad_unavail = [c for c in constants if c in unavailable]
    bad_typemm = [c for c in constants if c in type_mismatch]
    if bad_unavail:
        return False, {
            "template": template,
            "constants": constants,
            "unavailable_hits": bad_unavail,
            "type_mismatch_hits": bad_typemm,
            "reason": "references unavailable constant",
        }
    if bad_typemm:
        return False, {
            "template": template,
            "constants": constants,
            "unavailable_hits": bad_unavail,
            "type_mismatch_hits": bad_typemm,
            "reason": "references type-mismatch constant",
        }
    return True, {
        "template": template,
        "constants": constants,
        "unavailable_hits": [],
        "type_mismatch_hits": [],
        "reason": "",
    }


def filter_templates(
    templates: Iterable[str],
    unavailable: set[str] | None = None,
    type_mismatch: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Apply verify_template to a list. Returns (kept, dropped_diagnostics)."""
    kept: list[str] = []
    dropped: list[dict] = []
    for t in templates:
        ok, diag = verify_template(t, unavailable, type_mismatch)
        if ok:
            kept.append(t)
        else:
            dropped.append(diag)
    return kept, dropped


def verification_summary(
    templates: Iterable[str],
    unavailable: set[str] | None = None,
    type_mismatch: set[str] | None = None,
) -> dict:
    """High-level summary for reports. Includes constant census."""
    templates = list(templates)
    unavailable = unavailable if unavailable is not None else default_unavailable()
    type_mismatch = type_mismatch if type_mismatch is not None else default_type_mismatch()
    kept, dropped = filter_templates(templates, unavailable, type_mismatch)
    all_constants: set[str] = set()
    for t in templates:
        all_constants |= extract_template_constants(t)
    return {
        "template_count": len(templates),
        "template_constant_checked_count": len(all_constants),
        "template_constant_available_count": len(
            all_constants - unavailable - type_mismatch
        ),
        "template_constant_unavailable_count": len(all_constants & unavailable),
        "template_constant_type_mismatch_count": len(all_constants & type_mismatch),
        "filtered_template_count": len(dropped),
        "filtered_templates": [d["template"] for d in dropped],
        "filtered_template_constants": sorted(
            {c for d in dropped for c in (d["unavailable_hits"] + d["type_mismatch_hits"])}
        ),
        "constants_seen": sorted(all_constants),
    }


if __name__ == "__main__":
    # Smoke-test on v4.5's div family.
    sample = [
        "omega",
        "simp",
        "simp_all",
        "simp [Nat.div_eq_of_lt]",
        "simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]",
        "rw [Nat.div_eq_of_lt]",
        "rw [Nat.div_lt_iff_lt_mul']",
        "rw [Nat.div_lt_iff_lt_mul]",
        "rw [Nat.div_le_iff_le_mul]",
        "exact Nat.div_le_div_right ‹_›",
        "apply Nat.div_le_div_right",
        "simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
        "simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
        "simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]",
        "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        "constructor <;> intro h_split <;> omega",
        "constructor <;> intro h_split <;> simp_all",
        "induction {hyp_le} <;> simp_all",
        "induction {hyp_le} with | refl => exact Nat.le_refl _ | step h_step ih => exact ih.trans (Nat.div_le_succ_div _ _)",
    ]
    import json

    print(json.dumps(verification_summary(sample), indent=2, ensure_ascii=False))
