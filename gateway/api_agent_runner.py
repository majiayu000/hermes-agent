"""API-server agent execution helper.

Keeps request-scoped context binding out of the large API adapter file.
"""

from __future__ import annotations

import uuid
from typing import Any

from gateway.session_acl import has_principal_scope
from gateway.session_scope_store import (
    bind_session_scope,
    inherit_or_bind_session_scope,
    issue_or_refresh_sandbox_lease,
)


def run_agent_sync(
    adapter: Any,
    *,
    user_message: Any,
    conversation_history: list[dict[str, Any]],
    ephemeral_system_prompt: str | None = None,
    session_id: str | None = None,
    stream_delta_callback: Any = None,
    reasoning_callback: Any = None,
    tool_progress_callback: Any = None,
    tool_start_callback: Any = None,
    tool_complete_callback: Any = None,
    agent_ref: list | None = None,
    gateway_session_key: str | None = None,
    approval_session_key: str | None = None,
    approval_notify_callback: Any = None,
    prompt_session_key: str | None = None,
    prompt_notify_callback: Any = None,
    principal_scope: dict[str, Any] | None = None,
    agent_configurator: Any = None,
    agent_creation_overrides: dict[str, Any] | None = None,
    runtime_message_id: str | None = None,
    runtime_auxiliary_egress: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    from gateway.session_context import clear_session_vars, set_session_vars

    scope = dict(principal_scope or {})
    if has_principal_scope(scope) and session_id:
        db = adapter._ensure_session_db()
        if db is None:
            raise RuntimeError("Session database unavailable for scoped session binding")
        bind_session_scope(db, session_id, scope)
        lease = issue_or_refresh_sandbox_lease(db, session_id, scope)
        if lease:
            scope.setdefault("sandbox_id", lease["sandbox_id"])
            scope.setdefault("sandbox_status", lease["status"])
            scope.setdefault("sandbox_expires_at", lease["expires_at"])

    bind_api_session = getattr(adapter, "_bind_api_server_session", None)
    if bind_api_session is not None:
        tokens = bind_api_session(
            chat_id=session_id or "",
            session_key=gateway_session_key or session_id or "",
            session_id=session_id or "",
            principal_scope=scope,
        )
    else:
        tokens = set_session_vars(
            platform="api_server",
            chat_id=session_id or "",
            session_key=gateway_session_key or session_id or "",
            session_id=session_id or "",
            tenant_id=str(scope.get("tenant_id") or ""),
            workspace_id=str(scope.get("workspace_id") or ""),
            project_id=str(scope.get("project_id") or ""),
            user_id=str(scope.get("user_id") or ""),
            roles=scope.get("roles"),
            sandbox_id=str(scope.get("sandbox_id") or ""),
            sandbox_status=str(scope.get("sandbox_status") or "active"),
            sandbox_expires_at=scope.get("sandbox_expires_at"),
            async_delivery=False,
        )
    approval_token = None
    runtime_auxiliary_token = None
    prompt_callbacks_enabled = bool(prompt_session_key and prompt_notify_callback)

    def _prompt(kind: str, payload: dict[str, Any], *, timeout: float) -> str:
        from tools.clarify_gateway import register, wait_for_response

        request_id = f"{kind}_{uuid.uuid4().hex}"
        question = str(payload.get("question") or payload.get("prompt") or kind)
        choices = payload.get("choices")
        register(
            request_id,
            str(prompt_session_key or ""),
            question,
            list(choices) if isinstance(choices, list) else None,
        )
        notify_payload = dict(payload)
        notify_payload.update({"kind": kind, "request_id": request_id})
        prompt_notify_callback(notify_payload)
        return wait_for_response(request_id, timeout) or ""

    def _clarify_callback(question: str, choices=None) -> str:
        if not prompt_callbacks_enabled:
            return ""
        return _prompt(
            "clarify",
            {"question": question, "choices": list(choices) if choices else None},
            timeout=600,
        )

    def _sudo_callback() -> str:
        if not prompt_callbacks_enabled:
            return ""
        return _prompt("sudo", {"prompt": "Sudo password required"}, timeout=120)

    try:
        if runtime_auxiliary_egress:
            from agent.run_scoped_auxiliary import bind_run_scoped_auxiliary

            runtime_auxiliary_token = bind_run_scoped_auxiliary(
                runtime_auxiliary_egress
            )
        if approval_session_key:
            from tools.approval import (
                register_gateway_notify,
                reset_current_session_key,
                set_current_session_key,
                unregister_gateway_notify,
            )

            approval_token = set_current_session_key(approval_session_key)
            if approval_notify_callback is not None:
                register_gateway_notify(approval_session_key, approval_notify_callback)
        if prompt_callbacks_enabled:
            from tools.terminal_tool import set_sudo_password_callback

            set_sudo_password_callback(_sudo_callback)
        create_kwargs = dict(agent_creation_overrides or {})
        agent = adapter._create_agent(
            ephemeral_system_prompt=ephemeral_system_prompt,
            session_id=session_id,
            stream_delta_callback=stream_delta_callback,
            reasoning_callback=reasoning_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            gateway_session_key=gateway_session_key,
            clarify_callback=_clarify_callback if prompt_callbacks_enabled else None,
            **create_kwargs,
        )
        if agent_configurator is not None:
            agent_configurator(agent)
        if agent_ref is not None:
            agent_ref[0] = agent
        effective_task_id = session_id or str(uuid.uuid4())
        conversation_kwargs = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": effective_task_id,
        }
        if isinstance(runtime_message_id, str) and runtime_message_id.strip():
            conversation_kwargs["runtime_message_id"] = runtime_message_id
        result = agent.run_conversation(**conversation_kwargs)
        usage = {
            "input_tokens": getattr(agent, "session_input_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_output_tokens", 0) or 0,
            "cache_read_tokens": getattr(agent, "session_cache_read_tokens", 0) or 0,
            "cache_write_tokens": getattr(agent, "session_cache_write_tokens", 0) or 0,
            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            "api_calls": getattr(agent, "session_api_calls", 0) or 0,
        }
        effective_session_id = getattr(agent, "session_id", session_id)
        if isinstance(result, dict) and isinstance(effective_session_id, str) and effective_session_id:
            result["session_id"] = effective_session_id
        if has_principal_scope(scope):
            db = adapter._ensure_session_db()
            if db is None:
                raise RuntimeError("Session database unavailable for scoped session binding")
            if session_id:
                bind_session_scope(db, session_id, scope)
                issue_or_refresh_sandbox_lease(db, session_id, scope)
            if effective_session_id and effective_session_id != session_id:
                inherit_or_bind_session_scope(
                    db,
                    effective_session_id,
                    scope=scope,
                    parent_session_id=session_id,
                )
                issue_or_refresh_sandbox_lease(db, effective_session_id, scope)
        return result, usage
    finally:
        if runtime_auxiliary_token is not None:
            from agent.run_scoped_auxiliary import reset_run_scoped_auxiliary

            reset_run_scoped_auxiliary(runtime_auxiliary_token)
        from agent.auxiliary_client import clear_runtime_auxiliary_overrides

        clear_runtime_auxiliary_overrides()
        if prompt_callbacks_enabled:
            from tools.clarify_gateway import clear_session
            from tools.terminal_tool import set_sudo_password_callback

            clear_session(str(prompt_session_key or ""))
            set_sudo_password_callback(None)
        if approval_session_key:
            from tools.approval import reset_current_session_key, unregister_gateway_notify

            if approval_notify_callback is not None:
                unregister_gateway_notify(approval_session_key)
            if approval_token is not None:
                reset_current_session_key(approval_token)
        clear_session_vars(tokens)
