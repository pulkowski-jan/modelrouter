from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .classifier import classify, text_of
from .config import ADAPTIVE, VIRTUAL, Config
from .db import DB
from .routing import PREFIX, RouteInput, post_classify, pre_route

log = logging.getLogger("router")
DROP_RESP = {"content-encoding", "content-length", "transfer-encoding", "connection"}
WORKDIR_RE = re.compile(r"Working directory:\s*(\S.*)")


def conv_key(messages: list[dict]) -> str:
    sys_ = next((text_of(m.get("content")) for m in messages if m.get("role") in ("system", "developer")), "")
    usr = next((text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
    return hashlib.sha256((sys_ + "\x00" + usr).encode()).hexdigest()[:32]


def is_local_only(request: Request, messages: list[dict]) -> bool:
    if request.headers.get("x-router-local-only", "").lower() in ("1", "true", "yes"):
        return True
    for m in messages:
        if m.get("role") in ("system", "developer"):
            hit = WORKDIR_RE.search(text_of(m.get("content")))
            if hit and (Path(hit.group(1).strip()) / ".router-local-only").exists():
                return True
    return False


def find_prefix(messages: list[dict]) -> str | None:
    if not messages or messages[-1].get("role") != "user":
        return None
    first = text_of(messages[-1].get("content")).lstrip().split(None, 1)
    return first[0] if first and first[0] in PREFIX else None


def usage_from_line(line: bytes) -> dict | None:
    if not line.startswith(b"data:"):
        return None
    data = line[5:].strip()
    if not data or data == b"[DONE]":
        return None
    try:
        return json.loads(data).get("usage") or None
    except (ValueError, AttributeError):
        return None


def error(code: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "router_error"}}, status_code=code)


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="modelrouter")
    db = DB(cfg.db_path)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    @app.get("/v1/models")
    async def models():
        ids = [ADAPTIVE, *VIRTUAL]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "router"} for i in ids]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0 = time.perf_counter()
        body = await request.json()
        messages = body.get("messages") or []
        requested = body.get("model", ADAPTIVE)
        if requested != ADAPTIVE and requested not in VIRTUAL:
            return error(404, f"unknown model {requested!r}")

        key = conv_key(messages)
        last_role = messages[-1].get("role") if messages else None
        est = int(len(json.dumps(messages) + json.dumps(body.get("tools", []))) / cfg.chars_per_token)
        req = RouteInput(
            model=requested,
            prefix=find_prefix(messages),
            sticky_route=db.get_sticky(key, cfg.sticky_ttl_s) if last_role == "tool" else None,
            est_tokens=est,
            local_only=is_local_only(request, messages),
            budget_exhausted=db.spent_today() >= cfg.daily_budget_usd,
        )

        dec = pre_route(cfg, req)
        classifier_ms = None
        if dec.route is None and dec.error is None:
            verdict, classifier_ms = await classify(client, cfg, messages)
            if verdict is None:
                dec = post_classify(cfg, req, "strong_local", "classifier_fallback")
            else:
                dec = post_classify(cfg, req, verdict.route, verdict.reason_code)

        base = dict(conv_key=key, requested_model=requested, source=dec.source, reason=dec.reason,
                    classifier_ms=classifier_ms, est_prompt_tokens=est)
        if dec.error:
            code, msg = dec.error
            log.warning("rejected %s: %s", code, msg)
            db.log_request(**base, route=None, backend=None, status=code,
                           latency_ms=int((time.perf_counter() - t0) * 1000))
            return error(code, msg)

        route = dec.route
        if last_role != "tool":
            db.set_sticky(key, route)
        be = cfg.backend(route)
        remote = cfg.is_remote(route)
        log.info("route=%s source=%s reason=%s est=%d", route, dec.source, dec.reason, est)

        payload = dict(body, model=cfg.upstream_model(route))
        stream = bool(body.get("stream"))
        if stream:
            payload["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        headers = {"content-type": "application/json", "accept-encoding": "identity"}
        if remote:
            payload["usage"] = {"include": True}
            api_key = os.environ.get(be.api_key_env or "")
            if not api_key:
                return error(500, f"environment variable {be.api_key_env} is not set")
            headers["authorization"] = f"Bearer {api_key}"

        def record(status: int, usage: dict | None):
            usage = usage or {}
            db.log_request(**base, route=route, backend=cfg.spec(route), status=status,
                           latency_ms=int((time.perf_counter() - t0) * 1000),
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           cost_usd=usage.get("cost"))

        url = f"{be.base_url}/chat/completions"
        try:
            upstream_req = client.build_request("POST", url, json=payload, headers=headers)
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            record(502, None)
            return error(502, f"backend {cfg.spec(route)} unreachable: {e}")

        out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP_RESP}
        out_headers["x-router-route"] = route

        if not stream:
            raw = await upstream.aread()
            await upstream.aclose()
            try:
                usage = json.loads(raw).get("usage")
            except ValueError:
                usage = None
            record(upstream.status_code, usage)
            return JSONResponse(content=json.loads(raw) if raw else {}, status_code=upstream.status_code,
                                headers=out_headers) if usage is not None or raw[:1] == b"{" else error(502, "invalid upstream body")

        async def gen():
            buf, usage = b"", None
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for ln in lines:
                        usage = usage_from_line(ln.strip()) or usage
            finally:
                try:
                    await upstream.aclose()
                finally:
                    record(upstream.status_code, usage)

        return StreamingResponse(gen(), status_code=upstream.status_code, headers=out_headers,
                                 media_type=upstream.headers.get("content-type", "text/event-stream"))

    @app.on_event("shutdown")
    async def _close():
        await client.aclose()

    return app
