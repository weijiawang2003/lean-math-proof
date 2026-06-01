#!/usr/bin/env python3
"""RC4B shared gate + live disjoint_left-bridge probe (imported by the RC4B scripts).

The gate is the narrow disjoint_left_bridge policy. For a theorem, the gate fires when:
  * the namespace is Set or Multiset, AND
  * the name or goal/statement mentions disjointness ("disjoint" / "Disjoint").
When it fires it emits, for the matched namespace, the two bridge tactics
`simp [<NS>.disjoint_left]` and `simp [<NS>.disjoint_left] <;> aesop`. No match ->
no emission (cannot fire on Nat/List/Finset/Order goals, nor on Set/Multiset goals that
never mention disjointness — so Finset.disjoint_* and Nat/List negatives stay quiet).

The live probe (`run_tactics_live`) is reused verbatim from rc4a_gate to keep the
LeanDojo harness identical across RC4A / RC4B.
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


def namespace_of(namespace, full_name):
    """Authoritative namespace: explicit field if given, else the leading dotted
    segment of the full name."""
    ns = (namespace or "").strip()
    if ns:
        return ns
    return full_name.split(".")[0] if full_name else ""


def _mentions_disjoint(tokens, name, goal_text):
    blob = ((name or "") + " " + (goal_text or "")).lower()
    return any(t.lower() in blob for t in tokens)


def gate_fires(policy, namespace, goal_text, full_name):
    """Return (fires, bridge_namespace, tactics, action_names, lemma).

    `tactics` is the ordered list of gated bridge tactics to try (simp first, then
    `<;> aesop`). The matched namespace selects which actions apply; at most one
    namespace can match per theorem.
    """
    ns = namespace_of(namespace, full_name)
    matched_actions = []
    bridge_ns = None
    lemma = None
    for a in policy["actions"]:
        g = a["gate"]
        if g["requires_namespace"] != ns:
            continue
        if not _mentions_disjoint(g["requires_name_or_goal_contains"], full_name, goal_text):
            continue
        matched_actions.append(a)
        bridge_ns = ns
        lemma = a["lemma"]
    if not matched_actions:
        return False, None, [], [], None
    tactics = [a["tactic"] for a in matched_actions]
    action_names = [a["name"] for a in matched_actions]
    return True, bridge_ns, tactics, action_names, lemma


# ----------------------------- live probe ---------------------------------
class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def run_tactics_live(file_path, full_name, tactics, open_timeout=90, per_tactic=12):
    """Open one Dojo and run each tactic from the initial state. Returns
    {"live":bool,"setup_error":str|None,"ran":[{tactic,solved,outcome,error}]}.

    Reused verbatim from rc4a_gate so the LeanDojo harness is identical across RC4A/RC4B.
    """
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
