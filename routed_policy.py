"""NS13 — domain-aware routed generative policy.

Holds multiple ``GenerativePolicy`` instances and dispatches each
``rank_tactics`` / ``choose_tactic`` call to the one whose pattern
matches the theorem ``full_name``. A small JSON config defines the
routes; sub-checkpoints are loaded lazily on first hit.

Config schema::

    {
      "routes": [
        {"pattern": "^Nat\\.",    "ckpt_dir": "project/models/gen_v5_ns11_combined"},
        {"pattern": "^Set\\.",    "ckpt_dir": "project/models/gen_v5_ns12_balanced"},
        {"pattern": "^Finset\\.", "ckpt_dir": "project/models/gen_v5_ns12_balanced"}
      ],
      "default_ckpt_dir": "project/models/gen_v5_ns12_balanced"
    }

Routes are evaluated in order; the first matching pattern wins. If
no route matches, ``default_ckpt_dir`` is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from generative_policy import GenerativePolicy


class RoutedGenerativePolicy:
    """Domain-aware router over multiple ``GenerativePolicy`` instances."""

    def __init__(
        self,
        route_config: str,
        decode_mode: str = "beam",
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> None:
        cfg_path = Path(route_config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"route config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        routes = cfg.get("routes") or []
        if not routes:
            raise ValueError(f"route config has no routes: {cfg_path}")
        self._routes: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r["pattern"]), r["ckpt_dir"]) for r in routes
        ]
        self._default: str | None = (
            cfg.get("default_ckpt_dir") or cfg.get("default")
        )
        if self._default is None:
            raise ValueError(
                f"route config missing default_ckpt_dir: {cfg_path}"
            )

        self._decode_mode = decode_mode
        self._temperature = temperature
        self._seed = seed
        self._policies: dict[str, GenerativePolicy] = {}
        # Stats for end-of-run reporting.
        self.route_hits: dict[str, int] = {}

        print(
            f"RoutedGenerativePolicy: {len(self._routes)} routes, "
            f"default={self._default}"
        )
        for pat, ckpt in self._routes:
            print(f"  {pat.pattern!r} -> {ckpt}")

    def _ckpt_for(self, full_name: str) -> str:
        for pat, ckpt in self._routes:
            if pat.search(full_name):
                return ckpt
        return self._default  # type: ignore[return-value]

    def _policy_for(self, full_name: str) -> GenerativePolicy:
        ckpt = self._ckpt_for(full_name)
        self.route_hits[ckpt] = self.route_hits.get(ckpt, 0) + 1
        if ckpt not in self._policies:
            self._policies[ckpt] = GenerativePolicy(
                ckpt_dir=ckpt,
                decode_mode=self._decode_mode,
                temperature=self._temperature,
                seed=self._seed,
            )
        return self._policies[ckpt]

    @property
    def model_type(self) -> str:
        return "routed_generative"

    def generate_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        num_samples: int = 8,
        max_length: int = 128,
        temperature: float = 1.0,
        num_beams: int = 0,
    ) -> list[str]:
        return self._policy_for(full_name).generate_tactics(
            state_pp=state_pp,
            full_name=full_name,
            num_samples=num_samples,
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
        )

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 5,
    ) -> list[str]:
        return self._policy_for(full_name).rank_tactics(
            state_pp=state_pp, full_name=full_name, k=k
        )

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        return self._policy_for(full_name).choose_tactic(
            state_pp=state_pp, full_name=full_name
        )
