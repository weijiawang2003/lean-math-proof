#!/usr/bin/env python3
"""RC4A shared gate + live def-unfold probe (imported by the RC4A scripts).

The gate is the narrow def_unfold_simp policy: for a theorem, match the allowlisted
definitions whose name appears in the goal/statement/name; if any match, the candidate
emits a single `simp [<matched defs>]`. No match -> no emission (cannot fire on
Nat/arith/Multiset/List goals that name none of the allowlisted defs).
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACED_ROOT = os.path.expanduser(
    "~/.cache/lean_dojo/leanprover-community-mathlib4-"
    "29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4")


def load_policy(path):
    return json.load(open(path if os.path.isabs(path) else os.path.join(_REPO, path)))


def statement_from_source(file_path, full_name, root=_TRACED_ROOT):
    """Recover a declaration's signature (up to `:=`) from traced Mathlib source."""
    if not file_path or not root or not os.path.isdir(root):
        return None
    fp = os.path.join(root, file_path)
    if not os.path.exists(fp):
        return None
    short = full_name.split(".")[-1]
    pat = re.compile(r"^\s*(?:protected\s+|@\[[^\]]*\]\s*)*(?:theorem|lemma|def)\s+"
                     + re.escape(short) + r"\b")
    try:
        lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            buf = []
            for j in range(i, min(i + 14, len(lines))):
                buf.append(lines[j])
                if ":=" in lines[j]:
                    break
            text = " ".join(s.strip() for s in buf)
            idx = text.find(":=")
            if idx != -1:
                text = text[:idx]
            return re.sub(r"\s+", " ", text).strip()
    return None


def matched_defs(allowlist, goal_text, full_name):
    """Allowlisted defs whose name appears in the goal (identifier or .proj) or the
    theorem name. Deterministic order = allowlist order."""
    g = goal_text or ""
    idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*", g))
    idents |= {m[1:] for m in re.findall(r"\.[A-Za-z_][A-Za-z0-9_']+", g)}
    name_toks = set(re.split(r"[._]", full_name))
    out = []
    for d in allowlist:
        short = d.split(".")[-1]
        if d in idents or short in idents or short in name_toks:
            out.append(d)
    return out


def gate_fires(policy, goal_text, full_name):
    defs = matched_defs(policy["validated_def_allowlist"], goal_text, full_name)
    if not defs:
        return False, None, []
    tactic = "simp [" + ", ".join(defs) + "]"
    return True, tactic, defs


# ----------------------------- live probe ---------------------------------
class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def run_tactics_live(file_path, full_name, tactics, open_timeout=90, per_tactic=12):
    """Open one Dojo and run each tactic from the initial state. Returns
    {"live":bool,"setup_error":str|None,"ran":[{tactic,solved,outcome,error}]}."""
    res = {"live": False, "setup_error": None, "ran": []}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=file_path, full_name=full_name))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(open_timeout)
        try:
            dojo_cm = _Dojo(thm)
            dojo, state0 = dojo_cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        try:
            res["live"] = True
            for tac in tactics:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(per_tactic)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                    rec = getattr(out, "record", None)
                    fin = bool(getattr(out, "is_finished", False))
                    err = getattr(rec, "error_message", None) if rec else None
                    dead = bool(getattr(out, "session_dead", False))
                    r = {"tactic": tac, "solved": bool(fin),
                         "outcome": ("success" if fin else "proof_failed"),
                         "dead": bool(dead)}
                    if err and not fin:
                        r["error"] = err[:200]
                        el = err.lower()
                        if "unknown identifier" in el or "unknown constant" in el:
                            r["outcome"] = "unknown_name"
                        elif "unexpected token" in el or "unexpected identifier" in el:
                            r["outcome"] = "parse_error"
                except _ProbeTimeout:
                    r = {"tactic": tac, "solved": False, "outcome": "timeout", "dead": False}
                except Exception as e:
                    r = {"tactic": tac, "solved": False, "outcome": "exception",
                         "error": f"{type(e).__name__}: {str(e)[:120]}", "dead": False}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                res["ran"].append(r)
                if r["dead"]:
                    break
        finally:
            try:
                dojo_cm.__exit__(None, None, None)
            except Exception:
                pass
    except _ProbeTimeout:
        res["setup_error"] = f"dojo open exceeded {open_timeout}s"
    except Exception as e:
        import traceback
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:160]}\n" + traceback.format_exc()[-200:]
    return res
