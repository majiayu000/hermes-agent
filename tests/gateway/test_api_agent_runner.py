from gateway.api_agent_runner import run_agent_sync
from hermes_state import SessionDB


class _FakeAgent:
    session_prompt_tokens = 11
    session_completion_tokens = 2
    session_total_tokens = 13
    session_input_tokens = 3
    session_output_tokens = 2
    session_cache_read_tokens = 7
    session_cache_write_tokens = 1
    session_api_calls = 4

    def __init__(self, session_id):
        self.session_id = session_id

    def run_conversation(self, user_message, conversation_history, task_id):
        from agent.ultra_security import get_current_principal, get_current_sandbox_lease
        from gateway.session_context import get_session_env

        principal = get_current_principal()
        lease = get_current_sandbox_lease()
        return {
            "final_response": "ok",
            "observed": {
                "task_id": task_id,
                "session_id": get_session_env("HERMES_SESSION_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
                "tenant_id": principal.tenant_id if principal else "",
                "workspace_id": principal.workspace_id if principal else "",
                "project_id": principal.project_id if principal else "",
                "user_id": principal.user_id if principal else "",
                "roles": principal.roles if principal else (),
                "sandbox_id": lease.sandbox_id if lease else "",
            },
        }


class _FakeAdapter:
    def __init__(self, db):
        self._db = db

    def _create_agent(self, **kwargs):
        self.create_kwargs = kwargs
        return _FakeAgent(kwargs["session_id"])

    def _ensure_session_db(self):
        return self._db


def test_run_agent_sync_forwards_non_empty_runtime_message_id_only():
    class RuntimeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "runtime-session"

        def run_conversation(
            self,
            user_message,
            conversation_history,
            task_id,
            runtime_message_id,
        ):
            return {
                "final_response": "ok",
                "runtime_message_id": runtime_message_id,
            }

    class RuntimeAdapter:
        def _create_agent(self, **_kwargs):
            return RuntimeAgent()

    result, _usage = run_agent_sync(
        RuntimeAdapter(),
        user_message="runtime turn",
        conversation_history=[],
        session_id="runtime-session",
        runtime_message_id="wire-user-1",
    )

    assert result["runtime_message_id"] == "wire-user-1"


def test_run_agent_sync_binds_principal_scope_and_sandbox_lease(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    adapter = _FakeAdapter(db)

    try:
        result, usage = run_agent_sync(
            adapter,
            user_message="hello",
            conversation_history=[],
            session_id="session-1",
            gateway_session_key="channel-1",
            principal_scope={
                "tenant_id": "tenant-1",
                "workspace_id": "workspace-1",
                "project_id": "project-1",
                "user_id": "user-1",
                "roles": ("member",),
                "sandbox_id": "sandbox-1",
            },
        )
    finally:
        db.close()

    assert result["session_id"] == "session-1"
    assert result["observed"] == {
        "task_id": "session-1",
        "session_id": "session-1",
        "session_key": "channel-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "project_id": "project-1",
        "user_id": "user-1",
        "roles": ("member",),
        "sandbox_id": "sandbox-1",
    }
    assert usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_read_tokens": 7,
        "cache_write_tokens": 1,
        "total_tokens": 13,
        "api_calls": 4,
    }


def test_run_agent_sync_issues_persistent_sandbox_lease(tmp_path):
    from gateway.session_scope_store import get_sandbox_lease

    db = SessionDB(tmp_path / "state.db")
    adapter = _FakeAdapter(db)
    scope = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "project_id": "project-1",
        "user_id": "user-1",
        "roles": ("member",),
    }

    try:
        first, _ = run_agent_sync(
            adapter,
            user_message="hello",
            conversation_history=[],
            session_id="session-lease",
            principal_scope=scope,
        )
        lease = get_sandbox_lease(db, "session-lease")
        second, _ = run_agent_sync(
            adapter,
            user_message="again",
            conversation_history=[],
            session_id="session-lease",
            principal_scope=scope,
        )
    finally:
        db.close()

    assert first["observed"]["sandbox_id"].startswith("sbx_")
    assert lease is not None
    assert lease["sandbox_id"] == first["observed"]["sandbox_id"]
    assert lease["tenant_id"] == "tenant-1"
    assert lease["workspace_id"] == "workspace-1"
    assert lease["project_id"] == "project-1"
    assert lease["user_id"] == "user-1"
    assert lease["status"] == "active"
    assert second["observed"]["sandbox_id"] == first["observed"]["sandbox_id"]


def test_run_agent_sync_clears_context_after_turn(tmp_path):
    from agent.ultra_security import get_current_principal, get_current_sandbox_lease

    db = SessionDB(tmp_path / "state.db")
    try:
        run_agent_sync(
            _FakeAdapter(db),
            user_message="hello",
            conversation_history=[],
            session_id="session-1",
            principal_scope={
                "tenant_id": "tenant-1",
                "workspace_id": "workspace-1",
                "project_id": "project-1",
                "user_id": "user-1",
            },
        )
    finally:
        db.close()

    assert get_current_principal() is None
    assert get_current_sandbox_lease() is None


def test_run_agent_sync_binds_and_clears_runtime_auxiliary_egress():
    from agent.run_scoped_auxiliary import get_run_scoped_auxiliary

    class RuntimeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "runtime-session"

        def run_conversation(self, user_message, conversation_history, task_id):
            return {
                "final_response": "ok",
                "vision": get_run_scoped_auxiliary("vision"),
            }

    class RuntimeAdapter:
        def _create_agent(self, **_kwargs):
            return RuntimeAgent()

    capability = {
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
        "base_url": "http://agent-orchestrator:8093/internal/llm/v1",
        "grant": "ueg_" + "v" * 43,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    result, _usage = run_agent_sync(
        RuntimeAdapter(),
        user_message="runtime turn",
        conversation_history=[],
        session_id="runtime-session",
        runtime_auxiliary_egress={"vision": capability},
    )

    assert result["vision"] == capability
    assert get_run_scoped_auxiliary("vision") is None
