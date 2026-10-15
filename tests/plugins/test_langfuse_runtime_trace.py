"""Host run correlation and failure handling, without provider/network calls."""
import importlib
import sys

import pytest


class Span:
    def __init__(self, **fields):
        self.fields = fields
        self.children = []
        self.ended = False

    def update(self, **fields):
        self.fields.update(fields)

    def set_trace_io(self, **fields):
        self.fields.update(fields)

    def end(self):
        self.ended = True

    def start_observation(self, **fields):
        child = Span(**fields)
        self.children.append(child)
        return child

    def __enter__(self):
        return self


class Client:
    def __init__(self):
        self.roots = []

    def create_trace_id(self, *, seed):
        return seed

    def start_as_current_observation(self, **fields):
        root = Span(**fields)
        self.roots.append(root)
        return root

    def flush(self):
        pass


@pytest.fixture
def observer(monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home
    assert get_hermes_home().is_relative_to(tmp_path)
    sys.modules.pop("plugins.observability.langfuse", None)
    plugin = importlib.import_module("plugins.observability.langfuse")
    client = Client()
    monkeypatch.setattr(plugin, "_get_langfuse", lambda: client)
    monkeypatch.setattr(plugin, "propagate_attributes", None)
    return plugin, client


def start(plugin, *, turn="turn-1", run_id="run-1", count=1):
    ids = dict(session_id="session-1", task_id="session-1", turn_id=turn,
               api_request_id=f"{turn}:api:{count}", api_call_count=count)
    plugin.on_pre_llm_request(**ids, run_id=run_id, model="test-model",
                              request_messages=[{"role": "user", "content": "test"}])
    return ids


def test_product_run_groups_resume_but_separates_other_runs(observer):
    plugin, client = observer
    for turn, run_id in [("one", "run-1"), ("two", "run-1"), ("three", "run-2")]:
        ids = start(plugin, turn=turn, run_id=run_id)
        plugin.on_session_end(**ids, completed=True)
    traces = [root.fields["trace_context"]["trace_id"] for root in client.roots]
    assert traces[0] == traces[1] != traces[2]
    assert all(root.fields["metadata"]["completed"] for root in client.roots)
    for root in client.roots:
        assert root.children[0].fields["metadata"]["run_id"] in {"run-1", "run-2"}
        assert root.ended


def test_non_product_turns_do_not_merge_into_one_session_trace(observer):
    plugin, client = observer
    start(plugin, turn="one", run_id="")
    start(plugin, turn="two", run_id="")
    assert client.roots[0].fields["trace_context"] != client.roots[1].fields["trace_context"]


def test_failed_attempt_then_success_keeps_failure_and_closes_trace(observer):
    plugin, client = observer
    ids = start(plugin)
    plugin.on_api_request_error(**ids, error={"type": "TimeoutError", "message": "timed out"},
                                retryable=True, retry_count=1, status_code=504)
    first = client.roots[0].children[0]
    assert first.ended and first.fields["level"] == "ERROR"
    assert first.fields["metadata"]["status_code"] == 504
    assert not client.roots[0].ended
    ids = start(plugin, count=2)
    plugin.on_post_llm_call(**ids, assistant_content_chars=2, assistant_tool_call_count=0)
    assert len(client.roots) == 1
    assert client.roots[0].children[1].ended
    assert client.roots[0].ended
    assert not plugin._TRACE_STATE


@pytest.mark.parametrize("interrupted,completed,level", [(True, False, "WARNING"), (False, False, "ERROR")])
def test_incomplete_turn_closes_pending_children(observer, interrupted, completed, level):
    plugin, client = observer
    ids = start(plugin)
    plugin.on_pre_tool_call(**ids, tool_name="test_tool", tool_call_id="call-1", args={})
    plugin.on_session_end(**ids, interrupted=interrupted, completed=completed)
    root = client.roots[0]
    assert root.ended and root.fields["level"] == level
    assert all(child.ended for child in root.children)
    assert not plugin._TRACE_STATE


@pytest.mark.parametrize("status,level", [("error", "ERROR"), ("blocked", "WARNING"), ("cancelled", "WARNING")])
def test_tool_outcomes_are_visible(observer, status, level):
    plugin, client = observer
    ids = start(plugin)
    plugin.on_pre_tool_call(**ids, tool_name="test_tool", tool_call_id="call-1", args={})
    plugin.on_post_tool_call(**ids, tool_name="test_tool", tool_call_id="call-1",
                             result="failed", status=status, error_type="test_error")
    child = client.roots[0].children[-1]
    assert child.ended and child.fields["level"] == level
    assert child.fields["metadata"]["status"] == status


def test_telemetry_redacts_authorization_text(observer):
    plugin, _ = observer
    secret = "fake-secret-value-for-observer-test"
    assert secret not in plugin._safe_value(f"Authorization: Bearer {secret}")
    assert plugin._safe_value({"api_key": secret, "input_tokens": 4}) == {
        "api_key": "<redacted>", "input_tokens": 4,
    }


def test_run_context_reaches_hooks_and_is_reset_on_exception(monkeypatch):
    from gateway.api_agent_runner import run_agent_sync
    from hermes_cli.middleware import observer_run_id
    from hermes_cli.plugins import PluginManager
    manager = PluginManager()
    observed = []
    manager._hooks["pre_api_request"] = [lambda **data: observed.append(data)]

    class Agent:
        def run_conversation(self, **kwargs):
            manager.invoke_hook("pre_api_request")
            raise RuntimeError("test failure")

    class Adapter:
        def _create_agent(self, **kwargs):
            return Agent()

    for run_id in ["run-1", "run-2", ""]:
        with pytest.raises(RuntimeError, match="test failure"):
            run_agent_sync(Adapter(), user_message="test", conversation_history=[], runtime_run_id=run_id)
        assert observer_run_id.get() == ""
    assert [data.get("run_id", "") for data in observed] == ["run-1", "run-2", ""]


def test_concurrent_host_runs_keep_separate_observer_context():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from hermes_cli.middleware import observer_payload, observer_run_id
    barrier = Barrier(2)

    def observe(run_id):
        token = observer_run_id.set(run_id)
        try:
            barrier.wait(timeout=5)
            return observer_payload()["run_id"]
        finally:
            observer_run_id.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(observe, ["run-1", "run-2"])) == ["run-1", "run-2"]
    assert "run_id" not in observer_payload()


def test_observer_failure_does_not_interrupt_agent_hooks(observer, monkeypatch):
    from hermes_cli.plugins import PluginManager
    plugin, _ = observer
    manager = PluginManager()
    manager._hooks["pre_api_request"] = [plugin.on_pre_llm_request]
    def unavailable():
        raise RuntimeError("synthetic telemetry failure")
    monkeypatch.setattr(plugin, "_get_langfuse", unavailable)
    assert manager.invoke_hook("pre_api_request", session_id="test-session") == []


def test_late_error_after_cleanup_is_harmless(observer):
    plugin, _ = observer
    ids = start(plugin)
    plugin.on_session_end(**ids, completed=False)
    plugin.on_api_request_error(**ids, error={"type": "LateError"})
    plugin.on_session_end(**ids, completed=False)
    assert not plugin._TRACE_STATE
