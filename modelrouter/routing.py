from __future__ import annotations

from dataclasses import dataclass

from .config import VIRTUAL, Config

PREFIX = {"@remote": "remote_coding", "@local": "strong_local"}


@dataclass(frozen=True)
class RouteInput:
    model: str
    prefix: str | None
    sticky_route: str | None
    est_tokens: int
    local_only: bool
    budget_exhausted: bool


@dataclass(frozen=True)
class Decision:
    route: str | None
    reason: str
    source: str
    error: tuple[int, str] | None = None


def _err(code: int, msg: str, source: str) -> Decision:
    return Decision(None, msg, source, (code, msg))


def guard(cfg: Config, req: RouteInput, route: str, source: str, reason: str) -> Decision:
    explicit = source in ("forced", "prefix")
    remote = cfg.is_remote(route)
    if remote and req.local_only:
        if explicit or source == "sticky":
            return _err(403, "project is local-only; remote routes are forbidden", source)
        if source == "context":
            return _err(413, "prompt exceeds local context and project is local-only", source)
        return Decision("strong_local", "local_only_downgrade", "local_only")
    if remote and explicit and req.budget_exhausted:
        return _err(429, "daily remote budget exhausted", source)
    if not remote and source != "sticky":
        need = req.est_tokens + cfg.completion_reserve
        if need > cfg.context(route):
            if route == "fast_local" and not explicit and need <= cfg.context("strong_local"):
                return Decision("strong_local", "fast_context_bump", "context")
            return _err(413, f"prompt (~{req.est_tokens} tokens) exceeds {route} context", source)
    return Decision(route, reason, source)


def pre_route(cfg: Config, req: RouteInput) -> Decision:
    """Rules 1-4. route=None and no error means: call the classifier."""
    wanted = src = None
    if req.model in VIRTUAL:
        wanted, src = VIRTUAL[req.model], "forced"
    elif req.prefix:
        wanted, src = PREFIX[req.prefix], "prefix"
    if wanted:
        return guard(cfg, req, wanted, src, "explicit")
    if req.sticky_route:
        return guard(cfg, req, req.sticky_route, "sticky", "tool_loop")
    if req.est_tokens + cfg.completion_reserve > cfg.context("strong_local"):
        return guard(cfg, req, "remote_coding", "context", "prompt_exceeds_local_context")
    if req.budget_exhausted:
        return Decision("strong_local", "daily_budget_exhausted", "budget")
    return Decision(None, "", "")


def post_classify(cfg: Config, req: RouteInput, route: str, reason: str) -> Decision:
    return guard(cfg, req, route, "classifier", reason)
