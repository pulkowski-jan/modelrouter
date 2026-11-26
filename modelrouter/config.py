from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROUTES = ("fast_local", "strong_local", "remote_coding", "remote_reasoning")
VIRTUAL = {
    "local-fast": "fast_local",
    "local-strong": "strong_local",
    "remote-coding": "remote_coding",
    "remote-reasoning": "remote_reasoning",
}
ADAPTIVE = "adaptive"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class Backend:
    base_url: str
    context: int | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    db_path: str
    models: dict[str, str]
    backends: dict[str, Backend]
    daily_budget_usd: float
    classifier_timeout_s: float
    completion_reserve: int
    chars_per_token: float
    sticky_ttl_s: int

    def spec(self, route: str) -> str:
        return self.models[route]

    def is_remote(self, route: str) -> bool:
        return self.spec(route).startswith("openrouter/")

    def backend(self, route: str) -> Backend:
        spec = self.spec(route)
        return self.backends["openrouter" if spec.startswith("openrouter/") else spec]

    def upstream_model(self, route: str) -> str:
        spec = self.spec(route)
        return spec.removeprefix("openrouter/").removeprefix("local/")

    def context(self, route: str) -> int:
        return self.backend(route).context or 10**9


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("ROUTER_CONFIG", "config.yaml")
    raw = yaml.safe_load(Path(path).read_text())
    pol = raw.get("policy", {})
    cfg = Config(
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 8080)),
        db_path=raw.get("db_path", "router.sqlite"),
        models=raw["models"],
        backends={k: Backend(**v) for k, v in raw["backends"].items()},
        daily_budget_usd=float(pol.get("daily_budget_usd", 2.0)),
        classifier_timeout_s=float(pol.get("classifier_timeout_s", 3.0)),
        completion_reserve=int(pol.get("completion_reserve_tokens", 1024)),
        chars_per_token=float(pol.get("chars_per_token", 3.0)),
        sticky_ttl_s=int(pol.get("sticky_ttl_s", 3600)),
    )
    if cfg.host not in LOOPBACK:
        raise ValueError(f"refusing to bind to non-loopback host {cfg.host!r}")
    for r in (*ROUTES, "classifier"):
        if r not in cfg.models:
            raise ValueError(f"missing model for {r}")
        cfg.backend(r)
    return cfg
