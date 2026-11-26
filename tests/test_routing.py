import pytest

from modelrouter.config import Backend, Config
from modelrouter.routing import RouteInput, post_classify, pre_route


@pytest.fixture
def cfg():
    return Config(
        host="127.0.0.1", port=8080, db_path=":memory:",
        models={
            "classifier": "local/s", "fast_local": "local/s", "strong_local": "local/b",
            "remote_coding": "openrouter/qwen/qwen3-coder",
            "remote_reasoning": "openrouter/deepseek/deepseek-v3.2",
        },
        backends={
            "local/s": Backend("http://x", context=4096),
            "local/b": Backend("http://y", context=8192),
            "openrouter": Backend("http://z", api_key_env="K"),
        },
        daily_budget_usd=1.0, classifier_timeout_s=3.0, completion_reserve=1024,
        chars_per_token=3.0, sticky_ttl_s=60,
    )


def ri(**kw):
    d = dict(model="adaptive", prefix=None, sticky_route=None, est_tokens=100,
             local_only=False, budget_exhausted=False)
    d.update(kw)
    return RouteInput(**d)


def test_adaptive_goes_to_classifier(cfg):
    d = pre_route(cfg, ri())
    assert d.route is None and d.error is None


def test_forced_model(cfg):
    assert pre_route(cfg, ri(model="remote-coding")).route == "remote_coding"


def test_local_only_blocks_forced_remote(cfg):
    assert pre_route(cfg, ri(model="remote-coding", local_only=True)).error[0] == 403


def test_local_only_downgrades_classifier_remote(cfg):
    d = post_classify(cfg, ri(local_only=True), "remote_reasoning", "proof")
    assert d.route == "strong_local"


def test_sticky_reused(cfg):
    assert pre_route(cfg, ri(sticky_route="remote_coding")).route == "remote_coding"


def test_context_overflow_goes_remote(cfg):
    assert pre_route(cfg, ri(est_tokens=9000)).route == "remote_coding"


def test_context_overflow_local_only_rejected(cfg):
    assert pre_route(cfg, ri(est_tokens=9000, local_only=True)).error[0] == 413


def test_budget_exhausted_uses_strong_local(cfg):
    assert pre_route(cfg, ri(budget_exhausted=True)).route == "strong_local"


def test_forced_remote_rejected_when_budget_exhausted(cfg):
    assert pre_route(cfg, ri(model="remote-reasoning", budget_exhausted=True)).error[0] == 429


def test_fast_local_bumped_when_context_small(cfg):
    d = post_classify(cfg, ri(est_tokens=5000), "fast_local", "x")
    assert d.route == "strong_local"
