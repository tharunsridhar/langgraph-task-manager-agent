"""Tracing is opt-in and must never break a request. These tests make no network calls."""
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

import tracing
from agent import build_graph
from evals.run_evals import ScriptedModel, _fresh_db


def clean_env(monkeypatch, tmp_path):
    for var in ("OPIK_API_KEY", "OPIK_URL_OVERRIDE", "OPIK_TRACING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tracing.Path, "home", classmethod(lambda cls: tmp_path))


def test_off_by_default(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    assert tracing.tracing_enabled() is False
    assert tracing.get_callbacks("t1") == []


def test_enabled_by_key_or_config_file(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPIK_API_KEY", "x")
    assert tracing.tracing_enabled() is True
    monkeypatch.delenv("OPIK_API_KEY")
    (tmp_path / ".opik.config").write_text("[opik]\n")
    assert tracing.tracing_enabled() is True


def test_kill_switch(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPIK_API_KEY", "x")
    monkeypatch.setenv("OPIK_TRACING", "off")
    assert tracing.get_callbacks() == []


def test_tracer_failure_is_swallowed(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPIK_API_KEY", "x")
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name.startswith("opik"):
            raise ImportError("opik not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", broken)
    assert tracing.get_callbacks() == []


def test_app_adds_callbacks_only_when_enabled(monkeypatch, tmp_path):
    import app
    clean_env(monkeypatch, tmp_path)
    assert "callbacks" not in app.run_config_for("t")
    sentinel = object()
    monkeypatch.setattr(app.tracing, "get_callbacks", lambda thread_id=None: [sentinel])
    cfg = app.run_config_for("t")
    assert cfg["callbacks"] == [sentinel] and cfg["configurable"]["thread_id"] == "t"


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.names = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        if kwargs.get("name"):
            self.names.append(kwargs["name"])

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.names.append(f"tool:{serialized['name']}")


def test_graph_emits_a_span_per_node_and_tool():
    """What OpikTracer receives: the agent, guard, tools and quarantine nodes plus the tool call."""
    _fresh_db([{"title": "Gym"}])
    rec = Recorder()
    model = ScriptedModel([{"tool": "list_tasks", "args": {}}, {"say": "You have Gym."}])
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    graph.invoke({"messages": [{"role": "user", "content": "what is pending?"}]},
                 {"configurable": {"thread_id": "t"}, "callbacks": [rec]})
    for expected in ("agent", "guard", "tools", "quarantine", "tool:list_tasks"):
        assert expected in rec.names, rec.names
