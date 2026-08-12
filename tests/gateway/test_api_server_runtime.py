from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import gateway.api_server_runtime as runtime_module
from gateway.api_server_audit import request_audit_middleware
from gateway.api_server_shared import (
    MAX_REQUEST_BYTES,
    MAX_RUNTIME_ATTACHMENT_BYTES,
    MAX_RUNTIME_REQUEST_BYTES,
    body_limit_middleware,
)

from gateway.api_server_runtime import (
    APIServerRuntimeMixin,
    RuntimeBridgeSession,
    _allowed_skills_prompt,
    _failed_tool_result_projection,
    _normalize_runtime_messages,
    _pin_run_model,
    _replacement_system_prompt,
    _project_runtime_resume_attachments,
    _retry_session_db_history,
    _resume_session_db_history,
    _runtime_attachment_parts,
    _runtime_attachment_reference_prompt,
    _runtime_failure_code,
    _runtime_allowed_skill_digests,
    _runtime_image_paths,
    _runtime_allowed_skill_names,
    _runtime_skill_projections,
    _runtime_video_paths,
    _runtime_tool_middleware,
    _runtime_verified_activity_prompt,
    _validate_runtime_artifact_manifest,
)
from agent.tool_dispatch_helpers import DeferredToolResult
from model_tools import coerce_tool_args
from hermes_state import SessionDB
from gateway.runtime_contract import RUNTIME_DRIVER_FRAME_TYPES
from gateway.runtime_session_history import RuntimeSessionStateError, seed_runtime_session

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def test_replacement_system_prompt_logs_only_safe_digest_diagnostics(caplog):
    caplog.set_level(logging.ERROR, logger="gateway.api_server_runtime")

    with pytest.raises(ValueError, match="system_context digest mismatch"):
        _replacement_system_prompt({
            "version": " prompt-v1 ",
            "mode": " replace ",
            "digest": "sha256:bad",
            "stable": " trusted prompt ",
        })

    record = caplog.records[-1]
    rendered = record.getMessage()
    assert "received=sha256:bad" in rendered
    assert "expected=sha256:" in rendered
    assert "version_bytes=11/9" in rendered
    assert "mode_bytes=9/7" in rendered
    assert "stable_bytes=16/14" in rendered
    assert "trusted prompt" not in rendered


@pytest.mark.asyncio
async def test_runtime_run_body_limit_matches_inline_attachment_contract():
    assert MAX_RUNTIME_REQUEST_BYTES >= (
        MAX_REQUEST_BYTES
        + 4 * ((MAX_RUNTIME_ATTACHMENT_BYTES + 2) // 3)
    )

    async def consume(request):
        body = await request.read()
        return web.json_response({"bytes": len(body)})

    app = web.Application(
        middlewares=[body_limit_middleware],
        client_max_size=MAX_RUNTIME_REQUEST_BYTES,
    )
    app.router.add_post("/v1/runtime/runs", consume)
    app.router.add_post("/v1/responses", consume)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        payload = b"x" * (MAX_REQUEST_BYTES + 1)
        accepted = await client.post("/v1/runtime/runs", data=payload)
        assert accepted.status == 200
        assert await accepted.json() == {"bytes": len(payload)}

        rejected = await client.post("/v1/responses", data=payload)
        assert rejected.status == 413
        assert (await rejected.json())["error"]["code"] == "body_too_large"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ordinary_route_body_limit_rejects_chunked_bypass():
    async def consume(request):
        await request.read()
        return web.Response(status=204)

    async def oversized_chunks():
        chunk = b"x" * 1_000_000
        for _ in range(10):
            yield chunk
        yield b"x"

    app = web.Application(
        middlewares=[body_limit_middleware],
        client_max_size=MAX_RUNTIME_REQUEST_BYTES,
    )
    app.router.add_post("/v1/responses", consume)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/v1/responses", data=oversized_chunks())
        assert response.status == 413
        assert (await response.json())["error"]["code"] == "body_too_large"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_stream_emits_private_heartbeat_while_agent_is_quiet(monkeypatch):
    monkeypatch.setattr(runtime_module, "_RUNTIME_STREAM_HEARTBEAT_SECONDS", 0.01)
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    heartbeat = await runtime_module._next_runtime_stream_event(queue)
    assert heartbeat == {"type": "heartbeat", "payload": {}}

    event = {"type": "text_delta", "payload": {"delta": "ready"}}
    queue.put_nowait(event)
    assert await runtime_module._next_runtime_stream_event(queue) is event


class _MemorySessionDB:
    """Small SessionDB-shaped store for runtime handler tests."""

    def __init__(self):
        self.sessions = {}
        self.messages = {}

    def create_session(self, session_id, source, **kwargs):
        self.sessions.setdefault(session_id, {"id": session_id, "source": source, **kwargs})
        self.messages.setdefault(session_id, [])
        return session_id

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)
        return True

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages_as_conversation(self, session_id, include_ancestors=False):
        return copy.deepcopy(self.messages.get(session_id, []))

    def has_platform_message_id(self, session_id, platform_message_id):
        return any(
            message.get("message_id") == platform_message_id
            for message in self.messages.get(session_id, [])
        )

    def append_message(self, session_id, role, content=None, **fields):
        message = {"role": role, "content": copy.deepcopy(content)}
        message.update({key: copy.deepcopy(value) for key, value in fields.items() if value is not None})
        if fields.get("platform_message_id"):
            message["message_id"] = fields["platform_message_id"]
        self.messages.setdefault(session_id, []).append(message)
        return len(self.messages[session_id])


class _TestRuntimeAdapter(APIServerRuntimeMixin):
    def __init__(self):
        self.db = _MemorySessionDB()

    def _ensure_session_db(self):
        return self.db


def _runtime_call_db(session_id: str, *calls: tuple[str, str]) -> _MemorySessionDB:
    db = _MemorySessionDB()
    db.create_session(session_id, "api_server")
    db.append_message(
        session_id,
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for call_id, name in calls
        ],
    )
    return db


class _RuntimeAdapter(_TestRuntimeAdapter):
    def _check_auth(self, _request):
        return None

    async def _run_agent_bridge(self, **kwargs):
        agent = SimpleNamespace(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "skill_view", "description": "", "parameters": {"type": "object"}},
                },
                {
                    "type": "function",
                    "function": {"name": "skills_list", "description": "", "parameters": {"type": "object"}},
                },
                {
                    "type": "function",
                    "function": {"name": "web_search", "description": "", "parameters": {"type": "object"}},
                },
                {
                    "type": "function",
                    "function": {"name": "web_extract", "description": "", "parameters": {"type": "object"}},
                },
            ],
            valid_tool_names={"skill_view", "skills_list"},
            model="configured-model",
            _primary_runtime={
                "model": "configured-model",
                "compressor_model": "configured-model",
            },
            _fallback_chain=[{"provider": "other", "model": "fallback-model"}],
            _fallback_model={"provider": "other", "model": "fallback-model"},
            _fallback_index=0,
            _fallback_activated=False,
        )
        kwargs["agent_configurator"](agent)
        assert kwargs["ephemeral_system_prompt"] is None
        assert agent.model == "chat-test"
        assert agent._run_model_pin == "chat-test"
        assert agent._primary_runtime["model"] == "chat-test"
        assert agent._primary_runtime["compressor_model"] == "chat-test"
        assert agent._fallback_chain == []
        assert agent._fallback_model is None
        assert agent._skip_mcp_refresh is True
        assert agent._runtime_deferred_tool_names == {"ultra_media_job_create"}
        assert agent.valid_tool_names == {
            "ask_user_question",
            "image_analyze",
            "skill_view",
            "tool_search",
            "web_extract",
            "web_search",
        }
        ask_schema = next(
            tool["function"]["parameters"]
            for tool in agent.tools
            if tool["function"]["name"] == "ask_user_question"
        )
        option_schema = (
            ask_schema["properties"]["questions"]["items"]
            ["properties"]["options"]["items"]
        )
        assert option_schema["required"] == ["label", "value"]
        assert agent.ephemeral_system_prompt is None
        assert agent._cached_system_prompt == (
            "platform rules\n\ntrusted turn context\n\n"
            "<available_skills>\n"
            "- media-qa: Inspect generated media.\n"
            "- planning-only: Plan media without a delegated tool.\n"
            "</available_skills>"
        )
        assert agent._build_system_prompt() == agent._cached_system_prompt

        kwargs["tool_start_callback"]("skill_call", "skill_view", {
            "name": "media-qa",
            "task_id": "must-not-cross-runtime-boundary",
        })
        skill_result = _runtime_tool_middleware(
            tool_name="skill_view",
            args={"name": "media-qa"},
            session_id=kwargs["session_id"],
            tool_call_id="skill_call",
            next_call=lambda _args: pytest.fail("bound skill reached native skill_view"),
        )
        skill_envelope = json.loads(skill_result)
        assert skill_envelope["success"] is True
        assert "description: Inspect generated media." in skill_envelope["content"]
        assert skill_envelope["content"].endswith("workflow instructions\n")
        assert "skill_dir" not in skill_envelope
        assert skill_envelope["linked_files"] == {
            "references": ["references/guide.md"],
        }
        kwargs["tool_complete_callback"](
            "skill_call",
            "skill_view",
            {"name": "media-qa"},
            skill_result,
        )

        denied = _runtime_tool_middleware(
            tool_name="skill_view",
            args={"name": "tv-ad"},
            session_id=kwargs["session_id"],
            tool_call_id="draft_skill_call",
            next_call=lambda _args: pytest.fail("draft skill reached native skill_view"),
        )
        assert json.loads(denied) == {
            "success": False,
            "error": "Skill 'tv-ad' is not available for this run.",
        }

        unloaded = json.loads(_runtime_tool_middleware(
            tool_name="ultra_media_job_create",
            args={"operation": "image.generate", "prompt": "test"},
            session_id=kwargs["session_id"],
            tool_call_id="unloaded_01",
            next_call=lambda _args: pytest.fail("unloaded platform tool reached native dispatch"),
        ))
        assert unloaded["error"]["code"] == "tool_not_loaded"

        search_args = {"query": "create media"}
        kwargs["tool_start_callback"]("search_01", "tool_search", search_args)
        search_result = json.loads(_runtime_tool_middleware(
            tool_name="tool_search",
            args=search_args,
            session_id=kwargs["session_id"],
            tool_call_id="search_01",
        ))
        kwargs["tool_complete_callback"](
            "search_01",
            "tool_search",
            search_args,
            json.dumps(search_result),
        )
        assert [match["name"] for match in search_result["matches"]] == [
            "ultra_media_job_create",
        ]
        assert search_result["loaded_tools"] == ["ultra_media_job_create"]
        assert search_result["callable_on_next_step"] is True
        assert "ultra_media_job_create" in agent.valid_tool_names
        assert "tool_call" not in agent.valid_tool_names
        assert "tool_describe" not in agent.valid_tool_names

        delegated_args = {"operation": "image.generate", "prompt": "test"}
        self.db.append_message(
            kwargs["session_id"],
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "call_01",
                "type": "function",
                "function": {
                    "name": "ultra_media_job_create",
                    "arguments": json.dumps(delegated_args),
                },
            }],
        )
        kwargs["tool_start_callback"]("call_01", "ultra_media_job_create", delegated_args)
        tool_result = await asyncio.to_thread(
            _runtime_tool_middleware,
            tool_name="ultra_media_job_create",
            args=delegated_args,
            session_id=kwargs["session_id"],
            tool_call_id="call_01",
            next_call=lambda _args: pytest.fail("platform tool executed inside Hermes"),
        )
        assert json.loads(tool_result) == {"job_id": "job_01"}
        kwargs["tool_complete_callback"](
            "call_01",
            "ultra_media_job_create",
            delegated_args,
            tool_result,
        )
        return {"final_response": "asset://image/01"}, {"total_tokens": 3}


def test_pin_run_model_uses_canonical_switch_and_disables_fallbacks():
    calls = []

    def switch_model(model, provider, api_key, base_url, api_mode):
        calls.append((model, provider, api_key, base_url, api_mode))
        agent.model = model

    agent = SimpleNamespace(
        model="anthropic/claude-opus-4.8",
        provider="custom",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        api_mode="chat_completions",
        switch_model=switch_model,
        _primary_runtime={
            "model": "anthropic/claude-opus-4.8",
            "compressor_model": "anthropic/claude-opus-4.8",
        },
        _fallback_chain=[{"provider": "custom", "model": "zai-org/glm-5.2"}],
        _fallback_model={"provider": "custom", "model": "zai-org/glm-5.2"},
        _fallback_index=1,
        _fallback_activated=True,
    )

    pinned = _pin_run_model(agent, "anthropic/claude-opus-4.6")

    assert pinned == "anthropic/claude-opus-4.6"
    assert calls == [(
        "anthropic/claude-opus-4.6",
        "custom",
        "test-key",
        "https://example.invalid/v1",
        "chat_completions",
    )]
    assert agent.model == pinned
    assert agent._run_model_pin == pinned
    assert agent._primary_runtime["model"] == pinned
    assert agent._primary_runtime["compressor_model"] == pinned
    assert agent._fallback_chain == []
    assert agent._fallback_model is None
    assert agent._fallback_index == 0
    assert agent._fallback_activated is False


def test_runtime_attachment_parts_preserve_and_materialize_image_pixels(tmp_path):
    image = base64.b64encode(b"png-bytes").decode()
    parts = _runtime_attachment_parts([{
        "role": "product_photo",
        "asset_id": "asset_image",
        "filename": "product.png",
        "media_type": "image",
        "mime_type": "image/png",
        "data": image,
    }], image_dir=tmp_path)
    paths = _runtime_image_paths(parts)
    assert len(paths) == 1
    image_path = paths[0]
    assert image_path.parent == tmp_path
    assert image_path.name == hashlib.sha256(b"asset_image").hexdigest()[:24] + ".png"
    assert image_path.read_bytes() == b"png-bytes"
    assert parts == [{
        "type": "text",
        "text": (
            "[Attached image: product.png; role=product_photo; asset_id=asset_image. "
            "When pixel analysis is required, call image_analyze with "
            f"image_url={image_path}. Keep this private runtime path out "
            "of the final answer.]"
        ),
        "_runtime_reference_id": "asset_image",
        "_runtime_image_path": str(image_path),
    }, {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }]


def test_runtime_attachment_parts_require_private_image_directory():
    with pytest.raises(ValueError, match="image materialization directory"):
        _runtime_attachment_parts([{
            "role": "user_upload",
            "asset_id": "asset_image",
            "filename": "image.png",
            "media_type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(b"image-bytes").decode(),
        }])


def test_runtime_attachment_parts_materialize_generated_output_reference(tmp_path):
    image = base64.b64encode(b"generated-pixels").decode()
    parts = _runtime_attachment_parts([{
        "role": "generated_output",
        "reference_id": "output_board",
        "filename": "output_board.png",
        "media_type": "image",
        "mime_type": "image/png",
        "data": image,
    }], image_dir=tmp_path)
    image_path = _runtime_image_paths(parts)[0]
    assert image_path.name == hashlib.sha256(b"output_board").hexdigest()[:24] + ".png"
    assert image_path.read_bytes() == b"generated-pixels"
    assert "reference_id=output_board" in parts[0]["text"]
    assert "asset_id=" not in parts[0]["text"]


def test_runtime_attachment_parts_reject_mismatched_asset_and_reference_ids(tmp_path):
    with pytest.raises(ValueError, match="attachment identity"):
        _runtime_attachment_parts([{
            "role": "generated_output",
            "reference_id": "output_board",
            "asset_id": "asset_other",
            "filename": "output_board.png",
            "media_type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(b"generated-pixels").decode(),
        }], image_dir=tmp_path)



def test_runtime_attachment_parts_materialize_video_for_native_analysis(tmp_path):
    parts = _runtime_attachment_parts([{
        "role": "user_upload",
        "asset_id": "asset_video",
        "filename": "../../clip.mp4",
        "media_type": "video",
        "mime_type": "video/mp4",
        "data": base64.b64encode(b"video-bytes").decode(),
    }], video_dir=tmp_path)

    paths = _runtime_video_paths(parts)
    assert len(paths) == 1
    video_path = paths[0]
    assert video_path.parent == tmp_path
    assert video_path.name == hashlib.sha256(b"asset_video").hexdigest()[:24] + ".mp4"
    assert video_path.read_bytes() == b"video-bytes"
    assert parts == [{
        "type": "text",
        "text": (
            f"[Attached video: ../../clip.mp4; role=user_upload; asset_id=asset_video. "
            f"Analyze the complete source video with video_analyze using video_url={video_path} "
            "and include_transcript=true. "
            "Representative frames, when present, are supplementary rather than the source of truth.]"
        ),
        "_runtime_reference_id": "asset_video",
        "_runtime_video_path": str(video_path),
    }]


def test_runtime_attachment_parts_require_private_video_directory():
    with pytest.raises(ValueError, match="video materialization directory"):
        _runtime_attachment_parts([{
            "role": "user_upload",
            "asset_id": "asset_video",
            "filename": "clip.mp4",
            "media_type": "video",
            "mime_type": "video/mp4",
            "data": base64.b64encode(b"video-bytes").decode(),
        }])


def test_runtime_video_tool_is_scoped_to_materialized_attachment(tmp_path):
    allowed = tmp_path / "asset_video.mp4"
    allowed.write_bytes(b"video")
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_video",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_video",
        allowed_video_paths={str(allowed)},
        allowed_video_references={"output_video": str(allowed)},
    )
    runtime_module._SESSIONS["agent_video"] = session
    try:
        seen = []
        accepted = _runtime_tool_middleware(
            tool_name="video_analyze",
            args={"video_url": "output_video", "question": "Summarize it"},
            session_id="agent_video",
            tool_call_id="video_ok",
            next_call=lambda args: seen.append(args) or '{"success":true}',
        )
        assert accepted == '{"success":true}'
        assert seen == [{"video_url": str(allowed), "question": "Summarize it"}]

        denied = _runtime_tool_middleware(
            tool_name="video_analyze",
            args={"video_url": os.devnull, "question": "Read another file"},
            session_id="agent_video",
            tool_call_id="video_denied",
            next_call=lambda _args: pytest.fail("untrusted local path reached video tool"),
        )
        assert json.loads(denied) == {
            "success": False,
            "error": "video_analyze may only read video attachments owned by this run.",
            "error_code": "video_analysis_scope_denied",
            "retryable": False,
        }
    finally:
        runtime_module._SESSIONS.pop("agent_video", None)
        session.loop.close()


def test_runtime_video_tool_blocks_changed_retry_after_terminal_failure(tmp_path):
    allowed = tmp_path / "asset_video.mp4"
    allowed.write_bytes(b"video")
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_video_terminal",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_video_terminal",
        allowed_video_paths={str(allowed)},
    )
    halt_decisions = []
    session.agent_ref[0] = SimpleNamespace(
        _set_tool_guardrail_halt=halt_decisions.append,
    )
    runtime_module._SESSIONS["agent_video_terminal"] = session
    calls = []

    def fail_once(args):
        calls.append(args)
        return json.dumps({
            "success": False,
            "error": "upstream model rejected the request",
            "error_code": "video_analysis_model_incompatible",
            "retryable": False,
        })

    try:
        first = _runtime_tool_middleware(
            tool_name="video_analyze",
            args={"video_url": str(allowed), "question": "Summarize it"},
            session_id="agent_video_terminal",
            tool_call_id="video_first",
            next_call=fail_once,
        )
        assert json.loads(first)["error_code"] == "video_analysis_model_incompatible"

        second = _runtime_tool_middleware(
            tool_name="video_analyze",
            args={"video_url": str(allowed), "question": "Try a different prompt"},
            session_id="agent_video_terminal",
            tool_call_id="video_second",
            next_call=fail_once,
        )
        assert json.loads(second)["error"] == {
            "code": "repeated_non_retryable_tool_call",
            "message": (
                "Blocked video_analyze: an earlier call in this Run failed with "
                "non-retryable error video_analysis_model_incompatible."
            ),
            "retryable": False,
        }
        assert len(calls) == 1
        assert len(halt_decisions) == 1
        assert halt_decisions[0].code == "repeated_non_retryable_tool_call"
        assert halt_decisions[0].should_halt is True
    finally:
        runtime_module._SESSIONS.pop("agent_video_terminal", None)
        session.loop.close()


def test_runtime_image_tool_allows_remote_and_scopes_local_sources(tmp_path):
    allowed = tmp_path / "asset_image.png"
    allowed.write_bytes(b"image")
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_image",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_image",
        allowed_image_paths={str(allowed)},
        allowed_image_references={"output_image": str(allowed)},
    )
    runtime_module._SESSIONS["agent_image"] = session
    try:
        seen = []
        accepted = _runtime_tool_middleware(
            tool_name="image_analyze",
            args={
                "image_url": ["output_image", "https://example.com/reference.png"],
                "question": "Compare them",
            },
            session_id="agent_image",
            tool_call_id="image_ok",
            next_call=lambda args: seen.append(args) or '{"success":true}',
        )
        assert accepted == '{"success":true}'
        assert seen == [{
            "image_url": [str(allowed), "https://example.com/reference.png"],
            "question": "Compare them",
        }]

        denied = _runtime_tool_middleware(
            tool_name="image_analyze",
            args={"image_paths": os.devnull, "question": "Read another file"},
            session_id="agent_image",
            tool_call_id="image_denied",
            next_call=lambda _args: pytest.fail("untrusted local path reached image tool"),
        )
        assert json.loads(denied) == {
            "success": False,
            "error": (
                "image_analyze may only read HTTP(S) images or local image "
                "attachments owned by this run."
            ),
        }
    finally:
        runtime_module._SESSIONS.pop("agent_image", None)
        session.loop.close()


def test_runtime_image_tool_accepts_coerced_json_encoded_allowed_path_array(tmp_path):
    allowed_paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in allowed_paths:
        path.write_bytes(b"image")
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_image_json_array",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_image_json_array",
        allowed_image_paths={str(path) for path in allowed_paths},
    )
    runtime_module._SESSIONS["agent_image_json_array"] = session
    try:
        seen = []
        args = coerce_tool_args("image_analyze", {
            "image_paths": json.dumps([str(path) for path in allowed_paths]),
            "question": "Compare them",
        })
        assert args["image_paths"] == [str(path) for path in allowed_paths]
        accepted = _runtime_tool_middleware(
            tool_name="image_analyze",
            args=args,
            session_id="agent_image_json_array",
            tool_call_id="image_json_array_ok",
            next_call=lambda args: seen.append(args) or '{"success":true}',
        )
        assert accepted == '{"success":true}'
        assert seen == [{
            "image_paths": [str(path) for path in allowed_paths],
            "question": "Compare them",
        }]
    finally:
        runtime_module._SESSIONS.pop("agent_image_json_array", None)
        session.loop.close()


def test_runtime_image_tool_rejects_json_encoded_array_with_unowned_path(tmp_path):
    allowed = tmp_path / "allowed.png"
    unowned = tmp_path / "unowned.png"
    allowed.write_bytes(b"image")
    unowned.write_bytes(b"image")
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_image_json_array_denied",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_image_json_array_denied",
        allowed_image_paths={str(allowed)},
    )
    runtime_module._SESSIONS["agent_image_json_array_denied"] = session
    try:
        args = coerce_tool_args("image_analyze", {
            "image_paths": json.dumps([str(allowed), str(unowned)]),
            "question": "Compare them",
        })
        assert args["image_paths"] == [str(allowed), str(unowned)]
        denied = _runtime_tool_middleware(
            tool_name="image_analyze",
            args=args,
            session_id="agent_image_json_array_denied",
            tool_call_id="image_json_array_denied",
            next_call=lambda _args: pytest.fail(
                "an unowned path reached the image tool"
            ),
        )
        assert json.loads(denied) == {
            "success": False,
            "error": (
                "image_analyze may only read HTTP(S) images or local image "
                "attachments owned by this run."
            ),
        }
    finally:
        runtime_module._SESSIONS.pop("agent_image_json_array_denied", None)
        session.loop.close()


def test_runtime_tool_middleware_fails_closed_for_process_global_tools():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_scoped",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_scoped",
    )
    halt_decisions = []
    session.agent_ref[0] = SimpleNamespace(
        _set_tool_guardrail_halt=halt_decisions.append,
    )
    runtime_module._SESSIONS["agent_scoped"] = session
    try:
        denied = _runtime_tool_middleware(
            tool_name="mcp_higgsfield_media_upload",
            args={"filename": "reference.png"},
            session_id="agent_scoped",
            tool_call_id="unscoped_tool",
            next_call=lambda _args: pytest.fail(
                "process-global MCP tool escaped the Runtime scope"
            ),
        )
        assert json.loads(denied)["error"] == {
            "code": "tool_not_allowed",
            "message": (
                "Tool 'mcp_higgsfield_media_upload' is not authorized "
                "for this Runtime Run."
            ),
            "retryable": False,
        }
        assert len(halt_decisions) == 1
        assert halt_decisions[0].should_halt is True
        assert halt_decisions[0].code == "runtime_tool_scope_violation"
    finally:
        runtime_module._SESSIONS.pop("agent_scoped", None)
        session.loop.close()


def test_private_runtime_activity_arguments_are_never_exposed():
    assert runtime_module._activity_arguments("image_analyze", {
        "image_url": "output_board_123",
        "question": "check the layout",
    }) == {}


def test_private_runtime_activity_event_omits_arguments():
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_activity",
        loop,
        queue,
        [],
        1_000,
        "agent_activity",
    )
    emitted = []
    session.emit = lambda event_type, payload: emitted.append((event_type, payload))
    try:
        session.start_local_activity(
            "call_image_analyze",
            "image_analyze",
            {"image_url": "output_board_123", "question": "check the layout"},
        )
        assert emitted == [(
            "activity_started",
            {"call_id": "call_image_analyze", "name": "image_analyze"},
        )]
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_runtime_bridge_delivers_image_attachment_as_multimodal_user_content():
    captured = {}

    class AttachmentAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                _primary_runtime={
                    "model": "configured-model",
                    "compressor_model": "configured-model",
                },
                _fallback_chain=[],
                _fallback_model=None,
                _fallback_index=0,
                _fallback_activated=False,
            )
            kwargs["agent_configurator"](agent)
            captured["force_native_vision"] = getattr(
                agent,
                "_runtime_force_native_vision",
                False,
            )
            captured["tool_result_image_mode"] = (
                agent._runtime_tool_result_image_mode
            )
            captured["user_message"] = kwargs["user_message"]
            assert agent.valid_tool_names == {"image_analyze"}
            marker = next(
                part["text"]
                for part in kwargs["user_message"]
                if part.get("type") == "text" and "image_url=" in part.get("text", "")
            )
            image_path = marker.split("image_url=", 1)[1].split(". Keep", 1)[0]
            captured["image_path"] = image_path
            assert Path(image_path).read_bytes() == b"png-bytes"
            result = _runtime_tool_middleware(
                tool_name="image_analyze",
                args={"image_url": image_path, "question": "Describe it"},
                session_id=kwargs["session_id"],
                tool_call_id="image_call",
                next_call=lambda _args: '{"success":true,"analysis":"visible"}',
            )
            assert json.loads(result)["analysis"] == "visible"
            return {"final_response": "seen"}, {"total_tokens": 1}

    adapter = AttachmentAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        stable = "platform rules"
        version = "attachments/v1"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\nreplace\n{stable}".encode(),
        ).hexdigest()
        encoded = base64.b64encode(b"png-bytes").decode()
        response = await client.post("/v1/runtime/runs", json={
            "intent": "bootstrap",
            "run_id": "run_attachment",
            "model": "chat-test",
            "context": {"session_id": "session-run-attachment"},
            "messages": [{"id": "message-attachment", "role": "user", "content": "describe it"}],
            "system_context": {
                "version": version,
                "mode": "replace",
                "stable": stable,
                "digest": digest,
            },
            "attachments": [{
                "role": "user_upload",
                "asset_id": "asset_image",
                "filename": "reference.png",
                "media_type": "image",
                "mime_type": "image/png",
                "data": encoded,
            }],
        })
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        content = captured["user_message"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe it"}
        assert content[-1]["image_url"]["url"] == f"data:image/png;base64,{encoded}"
        assert captured["force_native_vision"] is False
        assert captured["tool_result_image_mode"] == "attach_by_ref"
        assert not Path(captured["image_path"]).exists()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_bridge_exposes_scoped_video_analysis_and_cleans_source_file():
    captured = {}

    class VideoAttachmentAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                _primary_runtime={
                    "model": "configured-model",
                    "compressor_model": "configured-model",
                },
                _fallback_chain=[],
                _fallback_model=None,
                _fallback_index=0,
                _fallback_activated=False,
            )
            kwargs["agent_configurator"](agent)
            assert agent.valid_tool_names == {"image_analyze", "video_analyze"}
            content = kwargs["user_message"]
            assert isinstance(content, list)
            marker = next(
                part["text"]
                for part in content
                if part.get("type") == "text" and "video_url=" in part.get("text", "")
            )
            video_path = marker.split("video_url=", 1)[1].split(".", 1)[0] + ".mp4"
            captured["video_path"] = video_path
            assert Path(video_path).read_bytes() == b"complete-video"
            result = _runtime_tool_middleware(
                tool_name="video_analyze",
                args={"video_url": video_path, "question": "Summarize it"},
                session_id=kwargs["session_id"],
                tool_call_id="video_call",
                next_call=lambda _args: '{"success":true,"analysis":"complete"}',
            )
            assert json.loads(result)["analysis"] == "complete"
            return {"final_response": "analyzed"}, {"total_tokens": 1}

    adapter = VideoAttachmentAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        stable = "platform rules"
        version = "video-attachments/v1"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\nreplace\n{stable}".encode(),
        ).hexdigest()
        response = await client.post("/v1/runtime/runs", json={
            "intent": "bootstrap",
            "run_id": "run_video_attachment",
            "model": "chat-test",
            "context": {"session_id": "session-run-video-attachment"},
            "messages": [{"id": "message-video-attachment", "role": "user", "content": "analyze the complete video"}],
            "system_context": {
                "version": version,
                "mode": "replace",
                "stable": stable,
                "digest": digest,
            },
            "attachments": [{
                "role": "user_upload",
                "asset_id": "asset_video",
                "filename": "source.mp4",
                "media_type": "video",
                "mime_type": "video/mp4",
                "data": base64.b64encode(b"complete-video").decode(),
            }],
        })
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        assert not Path(captured["video_path"]).exists()
    finally:
        await client.close()


def _runtime_skill_file(path: str, body: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_base64": base64.b64encode(body).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_runtime_driver_streams_tool_request_and_waits_for_result():
    adapter = _RuntimeAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    app.router.add_post("/v1/runtime/runs/{run_id}/tool-results", adapter._handle_runtime_tool_result)
    app.router.add_post("/v1/runtime/runs/{run_id}/interrupt", adapter._handle_runtime_interrupt)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        version = "ultrastudio-supercomputer/v1"
        mode = "replace"
        stable = "platform rules\n\ntrusted turn context"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\n{mode}\n{stable}".encode("utf-8"),
        ).hexdigest()
        package_digest = "sha256:" + hashlib.sha256(b"complete skill bundle").hexdigest()
        media_skill = b"""---
name: media-qa
kind: method
description: Inspect generated media.
---
workflow instructions
"""
        planning_skill = b"""---
name: planning-only
kind: method
description: Plan media without a delegated tool.
---
planning instructions
"""
        root_skill_digest = "sha256:" + hashlib.sha256(media_skill).hexdigest()
        response = await client.post("/v1/runtime/runs", json={
            "run_id": "run_test",
            "intent": "bootstrap",
            "model": "chat-test",
            "context": {"session_id": "panel_session_test"},
            "messages": [{"id": "message-run-test", "role": "user", "content": "make an image"}],
            "system_context": {
                "version": version,
                "mode": mode,
                "digest": digest,
                "stable": stable,
            },
            "skill_manifest": {
                "resolution_id": "resolution-test",
                "manifest_digest": "sha256:test",
                "skills": [
                    {
                        "runtime_alias": "media-qa",
                        "routing_mode": "primary",
                        "kind": "method",
                        "content_digest": package_digest,
                        "path": "/orchestrator-only/media-qa",
                        "files": [
                            _runtime_skill_file("SKILL.md", media_skill),
                            _runtime_skill_file(
                                "references/guide.md",
                                b"reference instructions",
                            ),
                        ],
                    },
                    {
                        "runtime_alias": "planning-only",
                        "routing_mode": "domain",
                        "kind": "method",
                        "content_digest": "sha256:" + hashlib.sha256(b"planning bundle").hexdigest(),
                        "path": "/orchestrator-only/planning-only",
                        "files": [
                            _runtime_skill_file("SKILL.md", planning_skill),
                        ],
                    },
                ],
            },
            "tools": [{
                "name": "ultra_media_job_create",
                "description": "create media",
                "input_schema": {"type": "object", "properties": {}},
                "exposure": "deferred",
                "route": "tokenrouter",
                "allowed_skills": ["media-qa"],
                "requires_skill_guidance": True,
            }, {
                "name": "ask_user_question",
                "description": "ask one structured question",
                "input_schema": {
                    "type": "object",
                    "required": ["questions"],
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["label", "value"],
                                            "properties": {
                                                "label": {"type": "string"},
                                                "value": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }],
        })
        assert response.status == 200
        started = json.loads(await response.content.readline())
        assert started["type"] == "run_started"
        assert started["payload"]["system_context_version"] == version
        assert started["payload"]["system_context_mode"] == "replace"
        assert started["payload"]["system_context_digest"] == digest
        activity_started = json.loads(await response.content.readline())
        assert activity_started == {
            "run_id": "run_test",
            "type": "activity_started",
            "payload": {
                "call_id": "skill_call",
                "name": "skill_view",
                "arguments": {"name": "media-qa"},
            },
        }
        activity_completed = json.loads(await response.content.readline())
        assert activity_completed == {
            "run_id": "run_test",
            "type": "activity_completed",
            "payload": {
                "call_id": "skill_call",
                "name": "skill_view",
                "status": "completed",
                "arguments": {
                    "digest": root_skill_digest,
                },
            },
        }
        search_started = json.loads(await response.content.readline())
        assert search_started == {
            "run_id": "run_test",
            "type": "activity_started",
            "payload": {
                "call_id": "search_01",
                "name": "tool_search",
                "arguments": {"query": "create media"},
            },
        }
        search_completed = json.loads(await response.content.readline())
        assert search_completed == {
            "run_id": "run_test",
            "type": "activity_completed",
            "payload": {
                "call_id": "search_01",
                "name": "tool_search",
                "status": "completed",
            },
        }
        tool_request = json.loads(await response.content.readline())
        assert tool_request["type"] == "tool_request"
        assert tool_request["payload"] == {
            "call_id": "call_01",
            "name": "ultra_media_job_create",
            "arguments": {"operation": "image.generate", "prompt": "test"},
        }

        delivered = await client.post("/v1/runtime/runs/run_test/tool-results", json={
            "call_id": "call_01",
            "ok": True,
            "result": {"job_id": "job_01"},
        })
        assert delivered.status == 204
        usage = json.loads(await response.content.readline())
        completed = json.loads(await response.content.readline())
        assert usage["type"] == "usage"
        assert completed["type"] == "completed"
        assert completed["payload"]["text"] == "asset://image/01"
    finally:
        await client.close()


def test_runtime_skill_manifest_is_visibility_source_of_truth():
    tool_digest = "sha256:" + "a" * 64
    planning_digest = "sha256:" + "b" * 64
    manifest = {
        "skills": [
            {"runtime_alias": "tool-guided", "content_digest": tool_digest},
            {"runtime_alias": "planning-only", "content_digest": planning_digest},
        ],
    }

    assert _runtime_allowed_skill_names(manifest) == {
        "tool-guided",
        "planning-only",
    }
    assert _runtime_allowed_skill_digests(manifest) == {
        "tool-guided": tool_digest,
        "planning-only": planning_digest,
    }
    assert _runtime_allowed_skill_names({
        "skills": [{"runtime_alias": "planning-only", "content_digest": planning_digest}],
    }) == {"planning-only"}
    with pytest.raises(ValueError, match="duplicate runtime_alias"):
        _runtime_allowed_skill_names({
            "skills": [
                {"runtime_alias": "duplicate", "content_digest": tool_digest},
                {"runtime_alias": "duplicate", "content_digest": planning_digest},
            ],
        })
    with pytest.raises(ValueError, match="content_digest is invalid for tool-guided"):
        _runtime_allowed_skill_digests({
            "skills": [{"runtime_alias": "tool-guided"}],
        })


def test_runtime_skill_projections_require_verified_inline_files():
    manifest = {
        "skills": [{
            "runtime_alias": "tool-guided",
            "routing_mode": "primary",
            "kind": "method",
            "content_digest": "sha256:" + "a" * 64,
            "path": "/orchestrator-only/tool-guided",
            "files": [_runtime_skill_file("SKILL.md", b"instructions")],
        }],
    }
    projections = _runtime_skill_projections(manifest)
    assert projections["tool-guided"].files == {"SKILL.md": b"instructions"}

    manifest["skills"][0]["files"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs from signed inventory for tool-guided"):
        _runtime_skill_projections(manifest)

    manifest["skills"][0]["files"] = [_runtime_skill_file("SKILL.md", b"instructions")]
    manifest["skills"][0]["files"][0]["content_base64"] = "not-base64!"
    with pytest.raises(ValueError, match="file content is invalid for tool-guided"):
        _runtime_skill_projections(manifest)

    manifest["skills"][0]["files"] = [_runtime_skill_file("../SKILL.md", b"outside")]
    with pytest.raises(ValueError, match="file path is invalid for tool-guided"):
        _runtime_skill_projections(manifest)

    manifest["skills"][0]["files"] = [_runtime_skill_file("guide.md", b"guide")]
    with pytest.raises(ValueError, match="root file is required for tool-guided"):
        _runtime_skill_projections(manifest)

    manifest["skills"][0].pop("files")
    with pytest.raises(ValueError, match="files are required for tool-guided"):
        _runtime_skill_projections(manifest)


def test_allowed_skill_prompt_uses_run_bound_projection_metadata():
    skill_body = b"""---
name: storyboard-quick-preview
kind: workflow
description: Legacy conversation-bound storyboard preview.
routing:
  mode: primary
  priority: 72
  triggers:
    - storyboard quick preview
    - quick campaign board video
    - legacy storyboard preview
  negative:
    - generic single video
---
instructions
"""
    manifest = {
        "skills": [{
            "runtime_alias": "storyboard-quick-preview",
            "routing_mode": "primary",
            "kind": "workflow",
            "content_digest": "sha256:" + hashlib.sha256(skill_body).hexdigest(),
            "files": [_runtime_skill_file("SKILL.md", skill_body)],
        }],
    }

    prompt = _allowed_skills_prompt(
        {"storyboard-quick-preview"},
        _runtime_skill_projections(manifest),
    )

    assert "- storyboard-quick-preview:" in prompt
    assert "priority=72" in prompt
    assert "applies=storyboard quick preview" in prompt


def test_dependency_only_and_malformed_skill_are_isolated_from_root_index(caplog):
    root_body = b"""---
name: root-flow
kind: method
description: Root workflow.
---
instructions
"""
    dependency_body = b"""---
name: helper-method
kind: method
description: Internal helper.
---
instructions
"""
    manifest = {
        "skills": [
            {
                "runtime_alias": "root-flow",
                "routing_mode": "primary",
                "kind": "method",
                "content_digest": "sha256:" + hashlib.sha256(root_body).hexdigest(),
                "files": [_runtime_skill_file("SKILL.md", root_body)],
            },
            {
                "runtime_alias": "helper-method",
                "routing_mode": "dependency_only",
                "kind": "method",
                "content_digest": "sha256:" + hashlib.sha256(dependency_body).hexdigest(),
                "files": [_runtime_skill_file("SKILL.md", dependency_body)],
            },
            {
                "runtime_alias": "broken-optional",
                "routing_mode": "domain",
                "kind": "method",
                "content_digest": "sha256:" + hashlib.sha256(b"\xff").hexdigest(),
                "files": [_runtime_skill_file("SKILL.md", b"\xff")],
            },
        ],
    }
    projections = _runtime_skill_projections(manifest)
    metadata = runtime_module.projection_skill_metadata(projections)
    prompt = _allowed_skills_prompt(
        {str(item["name"]) for item in metadata},
        metadata,
    )

    assert "- root-flow:" in prompt
    assert "helper-method" not in prompt
    assert "broken-optional" not in prompt
    assert "helper-method" in projections
    assert "Isolating unreadable run-bound Skill metadata" in caplog.text


def test_bound_skill_view_rejects_traversal_and_binary_files(monkeypatch):
    manifest = {
        "skills": [{
            "runtime_alias": "tool-guided",
            "routing_mode": "primary",
            "kind": "method",
            "content_digest": "sha256:" + "a" * 64,
            "path": "/orchestrator-only/tool-guided",
            "files": [
                _runtime_skill_file("SKILL.md", b"instructions"),
                _runtime_skill_file("guide.md", b"guide"),
                _runtime_skill_file("binary.dat", b"\xff"),
            ],
        }],
    }
    projections = _runtime_skill_projections(manifest)

    session_loop = asyncio.new_event_loop()
    session = RuntimeBridgeSession(
        "run_test",
        session_loop,
        asyncio.Queue(),
        [],
        0,
        "session_test",
        allowed_skill_names={"tool-guided"},
        allowed_skill_projections=projections,
    )
    monkeypatch.setitem(runtime_module._SESSIONS, "session_test", session)
    try:
        viewed = _runtime_tool_middleware(
            tool_name="skill_view",
            args={"name": "tool-guided", "file_path": "guide.md"},
            session_id="session_test",
            tool_call_id="guide_call",
            next_call=lambda _args: pytest.fail("bound skill reached native skill_view"),
        )
        assert json.loads(viewed)["content"] == "guide"

        traversal = _runtime_tool_middleware(
            tool_name="skill_view",
            args={"name": "tool-guided", "file_path": "../outside.md"},
            session_id="session_test",
            tool_call_id="traversal_call",
            next_call=lambda _args: pytest.fail("bound skill reached native skill_view"),
        )
        assert json.loads(traversal)["success"] is False

        binary = _runtime_tool_middleware(
            tool_name="skill_view",
            args={"name": "tool-guided", "file_path": "binary.dat"},
            session_id="session_test",
            tool_call_id="binary_call",
            next_call=lambda _args: pytest.fail("bound skill reached native skill_view"),
        )
        assert json.loads(binary)["success"] is False
    finally:
        session_loop.close()


def test_runtime_resume_projects_and_persists_result_without_synthetic_user():
    db = _MemorySessionDB()
    db.create_session("thread_resume", "api_server")
    db.append_message(
        "thread_resume",
        role="user",
        content="make an image",
        platform_message_id="user-1",
    )
    db.append_message(
        "thread_resume",
        role="assistant",
        content=None,
        tool_calls=[{
            "id": "call_media",
            "type": "function",
            "function": {
                "name": "media.generate_image",
                "arguments": '{"requests":[{"model":"image-model","prompt":"cat"}]}',
            },
        }],
        platform_message_id="assistant-1",
    )
    history = db.get_messages_as_conversation("thread_resume")
    result = {
        "tool_call_id": "call_media",
        "status": "succeeded",
        "output": {"batch_status": "succeeded", "jobs": [{"job_id": "job_1"}]},
    }
    resumed = _resume_session_db_history(db, "thread_resume", history, [result])
    assert [message["role"] for message in resumed] == ["user", "assistant", "tool"]
    assert resumed[-1]["tool_call_id"] == "call_media"
    assert json.loads(resumed[-1]["content"])["batch_status"] == "succeeded"
    assert sum(message["role"] == "user" for message in resumed) == 1


def test_runtime_resume_rejects_more_than_one_unfinished_tool_call():
    db = _MemorySessionDB()
    db.create_session("thread_resume_conflict", "api_server")
    db.append_message(
        "thread_resume_conflict",
        role="assistant",
        content=None,
        tool_calls=[
            {"id": "call_1", "function": {"name": "one", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "two", "arguments": "{}"}},
        ],
    )
    with pytest.raises(RuntimeSessionStateError, match="more than one unfinished"):
        _resume_session_db_history(
            db,
            "thread_resume_conflict",
            db.get_messages_as_conversation("thread_resume_conflict"),
            [{"tool_call_id": "call_1", "status": "succeeded", "output": {}}],
        )


def test_runtime_same_turn_retry_requires_user_or_tool_tail():
    history = [{"role": "user", "content": "make an image"}]
    assert _retry_session_db_history(history) == history
    with pytest.raises(RuntimeSessionStateError, match="must end with user or tool"):
        _retry_session_db_history(
            [*history, {"role": "assistant", "content": "finished"}],
        )


@pytest.mark.asyncio
async def test_runtime_resume_wiring_reaches_agent_without_new_user_message():
    class ResumeAdapter(_TestRuntimeAdapter):
        _api_key = ""

        def __init__(self):
            super().__init__()
            self.db.create_session("thread_session", "api_server")
            self.db.append_message(
                "thread_session",
                role="user",
                content="make an image",
                platform_message_id="user-resume",
            )
            self.db.append_message(
                "thread_session",
                role="assistant",
                content=None,
                tool_calls=[{
                    "id": "call_media",
                    "function": {"name": "media.generate_image", "arguments": "{}"},
                }],
                platform_message_id="assistant-resume",
            )

        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            assert agent._resume_from_tool_results is True
            assert "media.generate_image" in agent.valid_tool_names
            # Resume activation comes only from authoritative SessionDB tool
            # calls. This deferred tool remains discoverable via tool_search,
            # but no persisted call proves it was previously loaded or used.
            assert "platform.prompt_enhance" not in agent.valid_tool_names
            assert "platform.internal_reconcile" not in agent.valid_tool_names
            assert "tool_search" in agent.valid_tool_names
            assert kwargs["user_message"] == ""
            assert [message["role"] for message in kwargs["conversation_history"]] == [
                "user", "assistant", "tool",
            ]
            return {"final_response": "image complete"}, {"total_tokens": 2}

    adapter = ResumeAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        stable = "platform rules"
        version = "resume/v1"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\nreplace\n{stable}".encode(),
        ).hexdigest()
        response = await client.post("/v1/runtime/runs", json={
            "intent": "resume",
            "run_id": "run_resume",
            "model": "chat-test",
            "context": {"session_id": "thread_session"},
            "messages": [],
            "system_context": {
                "version": version,
                "mode": "replace",
                "stable": stable,
                "digest": digest,
            },
            "tool_results": [{
                "tool_call_id": "call_media",
                "status": "succeeded",
                "output": {"batch_status": "succeeded"},
            }],
            "tools": [{
                "name": "media.generate_image",
                "description": "generate an image",
                "input_schema": {"type": "object", "properties": {}},
                "exposure": "deferred",
            }, {
                "name": "platform.prompt_enhance",
                "description": "enhance a prompt",
                "input_schema": {"type": "object", "properties": {}},
                "exposure": "deferred",
            }, {
                "name": "platform.internal_reconcile",
                "description": "internal reconciliation",
                "input_schema": {"type": "object", "properties": {}},
                "exposure": "hidden",
            }],
        })
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        assert events[-1]["payload"]["text"] == "image complete"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_driver_reports_skill_failure_without_result_content():
    class FailingSkillAdapter(_RuntimeAdapter):
        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            kwargs["tool_start_callback"]("skill_failed", "skill_view", {
                "name": "missing-skill",
                "file_path": "SKILL.md",
            })
            kwargs["tool_complete_callback"](
                "skill_failed",
                "skill_view",
                {"name": "missing-skill"},
                json.dumps({
                    "success": False,
                    "error": "Skill 'missing-skill' not found.",
                    "available_skills": ["private-skill-name"],
                }),
            )
            return {"final_response": "could not load skill"}, {}

    adapter = FailingSkillAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        version = "ultrastudio-supercomputer/v1"
        mode = "replace"
        stable = "platform rules"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\n{mode}\n{stable}".encode("utf-8"),
        ).hexdigest()
        response = await client.post("/v1/runtime/runs", json={
            "run_id": "run_failed_skill",
            "intent": "bootstrap",
            "model": "chat-test",
            "context": {"session_id": "panel_session_failed_skill"},
            "messages": [{"id": "message-failed-skill", "role": "user", "content": "load a missing skill"}],
            "system_context": {
                "version": version,
                "mode": mode,
                "digest": digest,
                "stable": stable,
            },
        })
        events = [json.loads(line) async for line in response.content]
        completed = next(event for event in events if event["type"] == "activity_completed")
        assert completed["payload"] == {
            "call_id": "skill_failed",
            "name": "skill_view",
            "status": "failed",
            "error": {
                "code": "runtime_activity_failed",
                "message": "Skill 'missing-skill' not found.",
                "retryable": False,
            },
        }
        serialized = json.dumps(events)
        assert "private-skill-name" not in serialized
        assert "available_skills" not in serialized
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_bridge_blocks_unchanged_non_retryable_tool_retry():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_guard",
        asyncio.get_running_loop(),
        queue,
        [{"name": "ultra_prompt_compile", "input_schema": {"type": "object"}}],
        10_000,
        "agent_guard",
        _runtime_call_db("agent_guard", ("call_first", "ultra_prompt_compile")),
    )
    decisions = []
    session.agent_ref[0] = SimpleNamespace(
        _set_tool_guardrail_halt=decisions.append,
    )
    args = {"capability": "media.video.generate", "spec": {"intent": "ad"}}
    first = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "ultra_prompt_compile",
        args,
        "call_first",
    ))
    request = await queue.get()
    assert request["type"] == "tool_request"
    assert session.submit_result({
        "call_id": "call_first",
        "ok": False,
        "error": {
            "code": "invalid_tool_arguments",
            "message": "spec.prompt is required",
            "retryable": False,
        },
    })
    first_result = json.loads(await first)
    assert first_result["error"]["code"] == "invalid_tool_arguments"
    assert first_result["error"]["recovery"] == {
        "action": "correct_arguments",
        "remaining_attempts": 1,
        "same_arguments_allowed": False,
    }
    assert decisions == []

    second_result = json.loads(session.invoke_platform_tool(
        "ultra_prompt_compile",
        args,
        "call_second",
    ))
    assert second_result["error"]["code"] == "repeated_non_retryable_tool_call"
    assert decisions[0].code == "repeated_non_retryable_tool_call"
    assert queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["media.generate_image", "media.generate_audio"])
async def test_runtime_media_requires_exact_private_contract_before_submission(tool_name):
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_media_schema_required",
        asyncio.get_running_loop(),
        queue,
        [{"name": tool_name, "input_schema": {"type": "object"}}],
        10_000,
        "agent_media_schema_required",
        _runtime_call_db(
            "agent_media_schema_required",
            ("call_generate", tool_name),
        ),
    )
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        tool_name,
        {"requests": [{"model": "openai/gpt-image-2/text-to-image", "prompt": "poster"}]},
        "call_generate",
    ))
    request = await queue.get()
    assert request == {
        "run_id": "run_media_schema_required",
        "type": "runtime_control_request",
        "payload": {
            "request_id": request["payload"]["request_id"],
            "kind": "model_contract.get",
            "model": "openai/gpt-image-2/text-to-image",
        },
    }
    assert session.submit_control_result({
        "request_id": request["payload"]["request_id"],
        "ok": False,
        "error": {
            "code": "model_not_found",
            "message": "model schema unavailable",
            "retryable": False,
        },
    })
    result = json.loads(await call)
    assert result["error"]["code"] == "model_schema_unavailable"
    assert result["error"]["retryable"] is False
    assert queue.empty()


@pytest.mark.asyncio
async def test_runtime_media_uses_private_contract_and_rejects_domain_ratio_field():
    queue = asyncio.Queue()
    model = "openai/gpt-image-2/text-to-image"
    session = RuntimeBridgeSession(
        "run_media_schema",
        asyncio.get_running_loop(),
        queue,
        [{"name": "media.generate_image", "input_schema": {"type": "object"}}],
        10_000,
        "agent_media_schema",
        _runtime_call_db(
            "agent_media_schema",
            ("call_generate_bad", "media.generate_image"),
            ("call_generate_good", "media.generate_image"),
        ),
    )
    invalid_call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "media.generate_image",
        {"requests": [{"model": model, "prompt": "poster", "aspect_ratio": "3:2"}]},
        "call_generate_bad",
    ))
    contract_request = await queue.get()
    assert contract_request["type"] == "runtime_control_request"
    assert contract_request["payload"]["kind"] == "model_contract.get"
    assert contract_request["payload"]["model"] == model
    assert session.submit_control_result({
        "request_id": contract_request["payload"]["request_id"],
        "ok": True,
        "result": {
            "model": model,
            "observed_schema_digest": "sha256:" + "a" * 64,
            "parameters": [
                {"name": "prompt", "type": "string", "required": True},
                {
                    "name": "size",
                    "type": "string",
                    "required": False,
                    "options": ["1024x1024", "1536x1024"],
                    "description": "Arbitrary resolutions are supported as WIDTHxHEIGHT strings.",
                },
                {
                    "name": "quality",
                    "type": "string",
                    "required": False,
                    "options": ["low", "medium", "high"],
                },
            ],
        },
    })
    invalid = json.loads(await invalid_call)
    assert invalid["error"]["code"] == "invalid_tool_arguments"
    assert "aspect_ratio" in invalid["error"]["message"]
    assert queue.empty()

    generated = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "media.generate_image",
        {"requests": [{"model": model, "prompt": "poster", "size": "2304x1536"}]},
        "call_generate_good",
    ))
    request = await queue.get()
    assert request["payload"]["arguments"]["requests"][0]["size"] == "2304x1536"
    assert session.submit_result({
        "call_id": "call_generate_good",
        "ok": True,
        "result": {"delivery_status": "ready"},
    })
    assert json.loads(await generated) == {"delivery_status": "ready"}


@pytest.mark.asyncio
async def test_runtime_media_treats_platform_medias_as_required_provider_images():
    queue = asyncio.Queue()
    model = "bytedance/seedream-v5.0-pro/edit"
    session = RuntimeBridgeSession(
        "run_media_edit",
        asyncio.get_running_loop(),
        queue,
        [{"name": "media.generate_image", "input_schema": {"type": "object"}}],
        10_000,
        "agent_media_edit",
        _runtime_call_db(
            "agent_media_edit",
            ("call_generate_edit", "media.generate_image"),
        ),
    )
    args = {
        "requests": [{
            "model": model,
            "prompt": "Apply the locked style to @Image1",
            "medias": [{"role": "reference", "value": "output_character"}],
        }],
    }
    generated = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "media.generate_image",
        args,
        "call_generate_edit",
    ))
    contract_request = await queue.get()
    assert contract_request["type"] == "runtime_control_request"
    assert session.submit_control_result({
        "request_id": contract_request["payload"]["request_id"],
        "ok": True,
        "result": {
            "model": model,
            "observed_schema_digest": "sha256:" + "b" * 64,
            "parameters": [
                {"name": "prompt", "type": "string", "required": True},
                {"name": "images", "type": "array", "required": True},
            ],
        },
    })
    request = await queue.get()
    forwarded = request["payload"]["arguments"]["requests"][0]
    assert forwarded["medias"] == args["requests"][0]["medias"]
    assert "images" not in forwarded
    assert session.submit_result({
        "call_id": "call_generate_edit",
        "ok": True,
        "result": {"delivery_status": "ready"},
    })
    assert json.loads(await generated) == {"delivery_status": "ready"}


@pytest.mark.asyncio
async def test_runtime_bridge_allows_one_corrected_argument_attempt_then_halts():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_correction",
        asyncio.get_running_loop(),
        queue,
        [{"name": "ask_user_question", "input_schema": {"type": "object"}}],
        10_000,
        "agent_correction",
        _runtime_call_db(
            "agent_correction",
            ("call_invalid", "ask_user_question"),
            ("call_still_invalid", "ask_user_question"),
        ),
    )
    decisions = []
    agent = SimpleNamespace(_set_tool_guardrail_halt=decisions.append)
    session.agent_ref[0] = agent

    async def invoke(call_id, args, result):
        pending = asyncio.create_task(asyncio.to_thread(
            session.invoke_platform_tool,
            "ask_user_question",
            args,
            call_id,
        ))
        assert (await queue.get())["type"] == "tool_request"
        assert session.submit_result({"call_id": call_id, **result})
        return json.loads(await pending)

    first = await invoke(
        "call_invalid",
        {"questions": [{"options": [{"label": "A"}]}]},
        {
            "ok": False,
            "error": {
                "code": "invalid_tool_arguments",
                "message": "options[0].value is required",
                "retryable": False,
            },
        },
    )
    assert first["error"]["recovery"]["action"] == "correct_arguments"

    exhausted = await invoke(
        "call_still_invalid",
        {"questions": [{"options": [{"label": "A", "description": "retry"}]}]},
        {
            "ok": False,
            "error": {
                "code": "invalid_tool_arguments",
                "message": "options[0].value is required",
                "retryable": False,
            },
        },
    )
    assert exhausted["error"]["code"] == "argument_correction_exhausted"
    assert exhausted["error"]["cause"]["code"] == "invalid_tool_arguments"
    assert decisions[0].code == "argument_correction_exhausted"
    assert decisions[0].count == 2


@pytest.mark.asyncio
async def test_runtime_bridge_requires_persisted_tool_call_before_emitting_request():
    queue = asyncio.Queue()
    db = _MemorySessionDB()
    db.create_session("agent_unpersisted", "api_server")
    interrupted = []
    session = RuntimeBridgeSession(
        "run_unpersisted",
        asyncio.get_running_loop(),
        queue,
        [{"name": "platform.prompt_compile", "input_schema": {"type": "object"}}],
        10_000,
        "agent_unpersisted",
        db,
    )
    session.agent_ref[0] = SimpleNamespace(interrupt=interrupted.append)
    result = json.loads(await asyncio.to_thread(
        session.invoke_platform_tool,
        "platform.prompt_compile",
        {},
        "call_not_persisted",
    ))
    event = await queue.get()
    assert event["type"] == "error"
    assert event["payload"]["code"] == "runtime_history_conflict"
    assert result["error"]["code"] == "runtime_history_conflict"
    assert session.pending == {}
    assert interrupted


@pytest.mark.asyncio
async def test_runtime_bridge_preserves_safe_failed_result_with_typed_error():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_failed_projection",
        asyncio.get_running_loop(),
        queue,
        [{"name": "platform.prompt_compile", "input_schema": {"type": "object"}}],
        10_000,
        "agent_failed_projection",
        _runtime_call_db("agent_failed_projection", ("call_failed_projection", "platform.prompt_compile")),
    )
    session.agent_ref[0] = SimpleNamespace()
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "platform.prompt_compile",
        {},
        "call_failed_projection",
    ))
    assert (await queue.get())["type"] == "tool_request"
    assert session.submit_result({
        "call_id": "call_failed_projection",
        "ok": False,
        "result": {
            "allowed": {
                "aspect_ratios": ["16:9"],
                "durations": [5],
            },
        },
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "aspect ratio is not supported by model",
            "retryable": False,
        },
    })
    assert json.loads(await call) == {
        "result": {
            "allowed": {
                "aspect_ratios": ["16:9"],
                "durations": [5],
            },
        },
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "aspect ratio is not supported by model",
            "retryable": False,
        },
    }


@pytest.mark.parametrize("transport", [
    {
        "ok": False,
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "unsupported",
            "retryable": False,
            "private_upstream_detail": "must-not-cross",
        },
    },
    {
        "ok": False,
        "result": {"allowed": {"credential": "must-not-cross"}},
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "unsupported",
            "retryable": False,
        },
    },
    {
        "ok": False,
        "result": {"allowed": {"durations": [{"secret": "must-not-cross"}]}},
        "error": {
            "code": "unsupported_duration",
            "message": "unsupported",
            "retryable": False,
        },
    },
    {
        "ok": False,
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "unsupported",
        },
    },
    {
        "error": {
            "code": "unsupported_aspect_ratio",
            "message": "unsupported",
            "retryable": False,
        },
    },
])
def test_failed_tool_result_projection_fails_closed(transport):
    projected = _failed_tool_result_projection(transport)
    assert projected == {
        "error": {
            "code": "invalid_tool_result",
            "message": "tool failed with an invalid result envelope",
            "retryable": False,
        },
    }
    assert "must-not-cross" not in json.dumps(projected)


@pytest.mark.asyncio
async def test_runtime_bridge_preserves_terminal_platform_error_code():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_terminal",
        asyncio.get_running_loop(),
        queue,
        [{"name": "ultra_quota_snapshot", "input_schema": {"type": "object"}}],
        10_000,
        "agent_terminal",
        _runtime_call_db("agent_terminal", ("call_terminal", "ultra_quota_snapshot")),
    )
    decisions = []
    session.agent_ref[0] = SimpleNamespace(_set_tool_guardrail_halt=decisions.append)
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "ultra_quota_snapshot",
        {},
        "call_terminal",
    ))
    assert (await queue.get())["type"] == "tool_request"
    assert session.submit_result({
        "call_id": "call_terminal",
        "ok": False,
        "error": {
            "code": "tool_not_implemented",
            "message": "quota endpoint is not configured",
            "retryable": False,
        },
    })
    result = json.loads(await call)
    assert result["error"]["code"] == "tool_not_implemented"
    assert decisions[0].code == "tool_not_implemented"
    assert decisions[0].count == 1


def test_runtime_bridge_deadline_attribute_tracks_request():
    loop = asyncio.new_event_loop()
    try:
        unlimited = RuntimeBridgeSession("run_open", loop, asyncio.Queue(), [], 0, "agent_open")
        explicit = RuntimeBridgeSession("run_explicit", loop, asyncio.Queue(), [], 7_200_000, "agent_explicit")
        assert unlimited.deadline_seconds is None
        assert explicit.deadline_seconds == 7_200
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_runtime_driver_rejects_non_replacement_or_tampered_prompt():
    adapter = _RuntimeAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = {
            "run_id": "run_bad_prompt",
            "intent": "bootstrap",
            "model": "chat-test",
            "context": {"session_id": "panel_session_bad"},
            "messages": [{"id": "message-bad-prompt", "role": "user", "content": "hello"}],
            "system_context": {
                "version": "ultrastudio-supercomputer/v1",
                "mode": "append",
                "digest": "sha256:bad",
                "stable": "platform rules",
            },
        }
        response = await client.post("/v1/runtime/runs", json=body)
        assert response.status == 422
        payload = await response.json()
        assert "replacement" in payload["error"]["message"]

        body["system_context"]["mode"] = "replace"
        response = await client.post("/v1/runtime/runs", json=body)
        assert response.status == 422
        payload = await response.json()
        assert "digest mismatch" in payload["error"]["message"]
    finally:
        await client.close()


def _run_body(run_id: str, **extra):
    stable = "platform rules"
    version = "bridge-test/v1"
    digest = "sha256:" + hashlib.sha256(
        f"{version}\nreplace\n{stable}".encode("utf-8"),
    ).hexdigest()
    body = {
        "intent": "bootstrap",
        "run_id": run_id,
        "model": "chat-test",
        "context": {"session_id": f"session-{run_id}"},
        "messages": [{"id": f"message-{run_id}", "role": "user", "content": "go"}],
        "system_context": {
            "version": version,
            "mode": "replace",
            "stable": stable,
            "digest": digest,
        },
    }
    body.update(extra)
    if isinstance(body.get("messages"), list):
        body["messages"] = [
            {
                **message,
                "id": message.get("id") or f"message-{run_id}-{index}",
            }
            for index, message in enumerate(body["messages"])
        ]
    if body.get("intent") in {"resume", "retry"}:
        body["messages"] = []
    return body


def _complete_test_run(adapter, body):
    async def _run():
        app = web.Application()
        app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/v1/runtime/runs", json=body)
            status = response.status
            if response.content_type == "application/x-ndjson":
                payload = [json.loads(line) async for line in response.content]
            else:
                payload = await response.json()
            return status, payload
        finally:
            await client.close()

    return _run


@pytest.mark.asyncio
async def test_runtime_retry_continues_existing_turn_without_new_user_message():
    class RetryAdapter(_TestRuntimeAdapter):
        _api_key = ""

        def __init__(self):
            super().__init__()
            self.db.create_session("thread_retry", "api_server")
            self.db.append_message(
                "thread_retry",
                role="user",
                content="make an image",
                platform_message_id="user-retry",
            )

        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            assert agent._resume_from_tool_results is False
            assert agent._retry_current_turn is True
            assert kwargs["user_message"] == ""
            assert [message["role"] for message in kwargs["conversation_history"]] == ["user"]
            return {"final_response": "done"}, {"total_tokens": 2}

    status, events = await _complete_test_run(RetryAdapter(), _run_body(
        "retry",
        intent="retry",
        context={"session_id": "thread_retry"},
        retry_context={"attempt": 2, "previous_error_code": "provider_timeout"},
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"


def test_runtime_message_validation_requires_stable_unique_ids_and_valid_roles():
    with pytest.raises(ValueError, match="id must be a non-empty"):
        _normalize_runtime_messages([{"id": "", "role": "user", "content": "go"}])
    with pytest.raises(ValueError, match="unique"):
        _normalize_runtime_messages([
            {"id": "same", "role": "user", "content": "one"},
            {"id": "same", "role": "assistant", "content": "two"},
        ])
    with pytest.raises(ValueError, match="role is invalid"):
        _normalize_runtime_messages([{"id": "m1", "role": "system", "content": "no"}])
    with pytest.raises(ValueError, match="tool_call_id is required"):
        _normalize_runtime_messages([{"id": "m1", "role": "tool", "content": "result"}])


def test_runtime_typed_context_is_authenticated_and_never_message_content():
    messages = [{"message_id": "assistant-1", "role": "assistant", "content": "done"}]
    activity_prompt = _runtime_verified_activity_prompt({
        "verified_activities": [{
            "message_id": "assistant-1",
            "source_run_id": "source-run",
            "source_call_id": "source-call",
            "skill_name": "media-qa",
            "status": "completed",
            "file_path": "/orchestrator/reference.md",
            "digest": "sha256:" + "a" * 64,
        }],
    }, messages)
    reference_prompt = _runtime_attachment_reference_prompt({
        "generated_output": ["asset-output-1"],
        "user_upload": ["asset-upload-1"],
    })
    assert "assistant-1" in activity_prompt
    assert "asset-output-1" in reference_prompt
    assert all("source-run" not in message.get("content", "") for message in messages)
    with pytest.raises(ValueError, match="does not match an assistant"):
        _runtime_verified_activity_prompt({
            "verified_activities": [{
                "message_id": "forged",
                "source_run_id": "source-run",
                "source_call_id": "source-call",
                "skill_name": "media-qa",
                "status": "completed",
            }],
        }, messages)


def test_runtime_contract_removes_legacy_runtime_checkpoint_surface():
    assert "checkpoint" not in RUNTIME_DRIVER_FRAME_TYPES
    production = "\n".join(
        Path(path).read_text()
        for path in (
            runtime_module.__file__,
            str(Path(runtime_module.__file__).with_name("runtime_session_history.py")),
        )
    )
    for forbidden in (
        "runtime_checkpoint",
        "resume_runtime_history",
        "merge_runtime_session_history",
        "_anchor_content",
        "runtime_generated_media_context",
    ):
        assert forbidden not in production


@pytest.mark.asyncio
async def test_runtime_bootstrap_seeds_stable_ids_and_rejects_existing_session():
    class CaptureAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            self.captured = kwargs
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            return {"final_response": "done"}, {}

    adapter = CaptureAdapter()
    status, events = await _complete_test_run(adapter, _run_body(
        "run_bootstrap_ids",
        context={"session_id": "thread_bootstrap_ids"},
        messages=[
            {"id": "public-1", "role": "user", "content": "old"},
            {"id": "current-1", "role": "user", "content": "new"},
        ],
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"
    assert [message["message_id"] for message in adapter.db.messages["thread_bootstrap_ids"]] == [
        "public-1",
    ]
    assert adapter.captured["conversation_history"][0]["message_id"] == "public-1"
    assert adapter.captured["runtime_message_id"] == "current-1"

    adapter.db.create_session("thread_bootstrap_conflict", "api_server")
    status, payload = await _complete_test_run(
        adapter,
        _run_body(
            "run_bootstrap_conflict",
            context={"session_id": "thread_bootstrap_conflict"},
        ),
    )()
    assert status == 409
    assert payload["error"]["code"] == "runtime_session_conflict"


@pytest.mark.asyncio
async def test_runtime_new_turn_uses_only_current_id_and_rejects_missing_or_duplicate():
    class CaptureAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            self.captured = kwargs
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            return {"final_response": "done"}, {}

    adapter = CaptureAdapter()
    adapter.db.create_session("thread_new_turn", "api_server")
    adapter.db.append_message(
        "thread_new_turn",
        role="user",
        content="old",
        platform_message_id="old-id",
    )
    status, events = await _complete_test_run(adapter, _run_body(
        "run_new_turn",
        intent="new_turn",
        context={"session_id": "thread_new_turn"},
        messages=[{"id": "new-id", "role": "user", "content": "new"}],
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"
    assert [message["message_id"] for message in adapter.captured["conversation_history"]] == ["old-id"]
    assert adapter.captured["runtime_message_id"] == "new-id"

    status, payload = await _complete_test_run(
        adapter,
        _run_body(
            "run_new_turn_missing",
            intent="new_turn",
            context={"session_id": "missing-thread"},
            messages=[{"id": "new-id", "role": "user", "content": "new"}],
        ),
    )()
    assert status == 409
    assert payload["error"]["code"] == "runtime_session_not_found"

    status, payload = await _complete_test_run(
        adapter,
        _run_body(
            "run_new_turn_duplicate",
            intent="new_turn",
            context={"session_id": "thread_new_turn"},
            messages=[{"id": "old-id", "role": "user", "content": "changed content"}],
        ),
    )()
    assert status == 409
    assert payload["error"]["code"] == "runtime_message_id_conflict"


@pytest.mark.asyncio
async def test_runtime_rebootstrap_rebuilds_missing_new_turn_session():
    class CaptureAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            self.captured = kwargs
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            return {"final_response": "done"}, {}

    adapter = CaptureAdapter()
    status, events = await _complete_test_run(adapter, _run_body(
        "run_rebootstrap_new_turn",
        intent="rebootstrap",
        context={"session_id": "missing-new-turn"},
        messages=[
            {"id": "old-user", "role": "user", "content": "old"},
            {"id": "old-assistant", "role": "assistant", "content": "answer"},
            {"id": "current-user", "role": "user", "content": "continue"},
        ],
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"
    assert [item["message_id"] for item in adapter.captured["conversation_history"]] == [
        "old-user",
        "old-assistant",
    ]
    assert adapter.captured["runtime_message_id"] == "current-user"


@pytest.mark.asyncio
async def test_runtime_rebootstrap_restores_tool_result_without_reexecution():
    class CaptureAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            self.captured = kwargs
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            assert agent._resume_from_tool_results is True
            return {"final_response": "done"}, {}

    adapter = CaptureAdapter()
    result = {
        "tool_call_id": "call-media",
        "status": "succeeded",
        "output": {"asset_id": "asset-1"},
    }
    status, events = await _complete_test_run(adapter, _run_body(
        "run_rebootstrap_resume",
        intent="rebootstrap",
        context={"session_id": "missing-resume"},
        messages=[{"id": "current-user", "role": "user", "content": "create"}],
        tool_results=[result],
        recovery_tool_calls=[{
            "tool_call_id": "call-media",
            "tool_name": "media.generate_image",
            "args": {"model": "openai/gpt-image-2/text-to-image", "prompt": "x"},
        }],
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"
    history = adapter.captured["conversation_history"]
    assert [item["role"] for item in history] == ["user", "assistant", "tool"]
    assert history[-2]["tool_calls"][0]["id"] == "call-media"
    assert history[-1]["tool_call_id"] == "call-media"
    assert adapter.captured["runtime_message_id"] is None


@pytest.mark.asyncio
async def test_runtime_handler_accepts_bounded_artifact_manifest_without_message_projection():
    class ManifestAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            projected = json.dumps({
                "user_message": kwargs["user_message"],
                "history": kwargs["conversation_history"],
                "system": agent._cached_system_prompt,
            })
            assert "artifact-private-1" not in projected
            return {"final_response": "done"}, {}

    adapter = ManifestAdapter()
    status, events = await _complete_test_run(adapter, _run_body(
        "run_artifact_manifest",
        artifact_manifest=[{
            "tool_call_id": "call-private-1",
            "asset_id": "artifact-private-1",
            "media_type": "image",
            "role": "output",
            "request_index": 1,
            "output_index": 2,
            "source_run_id": "source-run-1",
            "event_seq": 17,
            "created_at": "2026-08-04T09:10:11.123456789Z",
        }],
    ))()
    assert status == 200
    assert events[-1]["type"] == "completed"

    status, payload = await _complete_test_run(
        adapter,
        _run_body(
            "run_artifact_manifest_invalid",
            context={"session_id": "thread-artifact-invalid"},
            artifact_manifest=[{
                "tool_call_id": "call-invalid",
                "asset_id": "asset-invalid",
                "media_type": "image",
                "role": "output",
                "source_run_id": "source-run",
                "event_seq": 1,
                "created_at": "2026-08-04T09:10:11Z",
                "unexpected": True,
            }],
        ),
    )()
    assert status == 422
    assert payload["error"]["code"] == "invalid_param"

    status, payload = await _complete_test_run(
        adapter,
        _run_body(
            "run_artifact_manifest_unbounded",
            context={"session_id": "thread-artifact-unbounded"},
            artifact_manifest=[{
                "tool_call_id": f"call-{index}",
                "asset_id": f"asset-{index}",
                "media_type": "image",
                "role": "output",
                "source_run_id": "source-run",
                "event_seq": index,
                "created_at": "2026-08-04T09:10:11Z",
            } for index in range(33)],
        ),
    )()
    assert status == 422
    assert "more than 32 entries" in payload["error"]["message"]


def test_artifact_manifest_validator_is_bounded_and_typed():
    go_serialized = [{
        "tool_call_id": "call-1",
        "asset_id": "asset-1",
        "media_type": "image",
        "role": "output",
        "request_index": 3,
        "output_index": 1,
        "source_run_id": "run-source-1",
        "event_seq": 42,
        "created_at": "2026-08-04T09:10:11.123456789Z",
    }]
    _validate_runtime_artifact_manifest(go_serialized)
    without_omitempty_indexes = dict(go_serialized[0])
    without_omitempty_indexes.pop("request_index")
    without_omitempty_indexes.pop("output_index")
    _validate_runtime_artifact_manifest([without_omitempty_indexes])
    _validate_runtime_artifact_manifest([])
    _validate_runtime_artifact_manifest(None)

    with pytest.raises(ValueError, match="must be an array"):
        _validate_runtime_artifact_manifest({})
    with pytest.raises(ValueError, match="invalid fields"):
        _validate_runtime_artifact_manifest([{**go_serialized[0], "url": "private"}])
    with pytest.raises(ValueError, match="asset_id must be non-empty"):
        _validate_runtime_artifact_manifest([{**go_serialized[0], "asset_id": ""}])
    with pytest.raises(ValueError, match="media_type is invalid"):
        _validate_runtime_artifact_manifest([{**go_serialized[0], "media_type": 7}])
    with pytest.raises(ValueError, match="event_seq must be a non-negative integer"):
        _validate_runtime_artifact_manifest([{**go_serialized[0], "event_seq": True}])
    with pytest.raises(ValueError, match="output_index must be a non-negative integer"):
        _validate_runtime_artifact_manifest([{**go_serialized[0], "output_index": -1}])
    with pytest.raises(ValueError, match="created_at must be RFC3339"):
        _validate_runtime_artifact_manifest([{
            **go_serialized[0],
            "created_at": "2026-08-04 09:10:11",
        }])
    with pytest.raises(ValueError, match="more than 32 entries"):
        _validate_runtime_artifact_manifest(go_serialized * 33)


def test_resume_attachment_projection_strips_private_runtime_metadata():
    durable_history = [{
        "role": "tool",
        "tool_call_id": "call-media",
        "content": '{"asset_id":"asset-1"}',
    }]
    private_parts = [
        {
            "type": "text",
            "text": "[Attached image: output.png; image_url=/tmp/output.png. Keep it scoped.]",
            "_runtime_image_path": "/tmp/output.png",
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cGl4ZWxz"},
        },
    ]

    projected = _project_runtime_resume_attachments(durable_history, private_parts)

    assert projected is not durable_history
    assert durable_history[-1]["content"] == '{"asset_id":"asset-1"}'
    assert isinstance(projected[-1]["content"], str)
    assert "_runtime_image_path" not in projected[-1]["content"]
    assert "data:image" not in projected[-1]["content"]
    assert "image_url=/tmp/output.png" in projected[-1]["content"]


def test_seed_runtime_session_reports_seed_and_cleanup_failures():
    class FailingSeedDB:
        def create_session(self, **_kwargs):
            return None

        def append_message(self, **_kwargs):
            raise OSError("seed write failed")

        def delete_session(self, _session_id):
            raise OSError("cleanup delete failed")

    with pytest.raises(RuntimeSessionStateError) as failure:
        seed_runtime_session(
            FailingSeedDB(),
            "thread-seed-failure",
            model="test-model",
            system_prompt="system",
            messages=[{
                "message_id": "wire-1",
                "role": "user",
                "content": "hello",
            }],
        )

    cause = failure.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert [str(error) for error in cause.exceptions] == [
        "seed write failed",
        "cleanup delete failed",
    ]


@pytest.mark.asyncio
async def test_runtime_run_over_limit_returns_retryable_429(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_MAX_CONCURRENT", "1")
    release = asyncio.Event()

    class BlockingAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            await release.wait()
            return {"final_response": "done"}, {}

    adapter = BlockingAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        first = await client.post("/v1/runtime/runs", json=_run_body("run_gate_a"))
        assert first.status == 200
        assert json.loads(await first.content.readline())["type"] == "run_started"

        second = await client.post("/v1/runtime/runs", json=_run_body("run_gate_b"))
        assert second.status == 429
        payload = await second.json()
        assert payload["error"]["code"] == "runtime_concurrency_exceeded"
        assert payload["error"]["retryable"] is True

        release.set()
        events = [json.loads(line) async for line in first.content]
        assert events[-1]["type"] == "completed"

        active = -1
        for _ in range(200):
            with runtime_module._RUNTIME_GATE_LOCK:
                active = runtime_module._ACTIVE_RUN_COUNT
            if active == 0:
                break
            await asyncio.sleep(0.01)
        assert active == 0
    finally:
        release.set()
        await client.close()


@pytest.mark.asyncio
async def test_runtime_interrupt_waits_without_thread_pool(monkeypatch):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("interrupt path must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", _forbidden)

    class InterruptAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            stop = asyncio.Event()
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                interrupt=lambda reason: stop.set(),
            )
            kwargs["agent_configurator"](agent)
            await stop.wait()
            return {"final_response": "stopped"}, {}

    adapter = InterruptAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    app.router.add_post("/v1/runtime/runs/{run_id}/interrupt", adapter._handle_runtime_interrupt)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        run_response = await client.post("/v1/runtime/runs", json=_run_body("run_interrupt_async"))
        assert run_response.status == 200
        assert json.loads(await run_response.content.readline())["type"] == "run_started"

        interrupt_response = await client.post(
            "/v1/runtime/runs/run_interrupt_async/interrupt",
            json={"reason": "user cancelled"},
        )
        assert interrupt_response.status == 204

        events = [json.loads(line) async for line in run_response.content]
        assert events[-1]["type"] == "completed"
        assert events[-1]["payload"]["text"] == "stopped"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_disconnect_interrupts_run_and_clears_session():
    class DisconnectAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            stop = asyncio.Event()
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                interrupt=lambda reason: stop.set(),
            )
            kwargs["agent_configurator"](agent)
            for index in range(2000):
                if stop.is_set():
                    break
                kwargs["stream_delta_callback"](f"delta-{index}")
                await asyncio.sleep(0.01)
            assert stop.is_set(), "pump never interrupted the run after disconnect"
            return {"final_response": "stopped"}, {}

    adapter = DisconnectAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        run_response = await client.post("/v1/runtime/runs", json=_run_body("run_disconnect"))
        assert run_response.status == 200
        assert json.loads(await run_response.content.readline())["type"] == "run_started"
        with runtime_module._SESSIONS_LOCK:
            session = runtime_module._SESSIONS.get("run_disconnect")
        assert session is not None

        run_response.close()

        for _ in range(1000):
            with runtime_module._SESSIONS_LOCK:
                cleared = "run_disconnect" not in runtime_module._SESSIONS
            if cleared and session.finished.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("run kept going after orchestrator disconnect")
        assert session.interrupted.is_set()
    finally:
        await client.close()


def test_resume_history_projects_externalized_output_ref():
    db = _runtime_call_db("thread_ext", ("call_ext", "media.generate_image"))
    history = _resume_session_db_history(
        db,
        "thread_ext",
        db.get_messages_as_conversation("thread_ext"),
        [{
            "tool_call_id": "call_ext",
            "status": "succeeded",
            "output_ref": "asset://image/01",
        }],
    )
    assert json.loads(history[-1]["content"]) == {
        "status": "externalized",
        "output_ref": "asset://image/01",
    }


def test_resume_history_rejects_ambiguous_output_fields():
    db = _runtime_call_db("thread_inline", ("call_inline", "media.generate_image"))
    with pytest.raises(ValueError, match="exactly one output field"):
        _resume_session_db_history(
            db,
            "thread_inline",
            db.get_messages_as_conversation("thread_inline"),
            [{
                "tool_call_id": "call_inline",
                "status": "succeeded",
                "output": {"url": "asset://image/1"},
                "output_ref": "asset://image/01",
            }],
        )


@pytest.mark.asyncio
async def test_unbounded_deadline_tool_wait_is_capped(monkeypatch):
    assert runtime_module._UNBOUNDED_TOOL_WAIT_CAP_SECONDS == 3600.0
    monkeypatch.setattr(runtime_module, "_UNBOUNDED_TOOL_WAIT_CAP_SECONDS", 0.05)
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_cap",
        asyncio.get_running_loop(),
        queue,
        [{"name": "ultra_media_job_create", "input_schema": {"type": "object"}}],
        0,
        "agent_cap",
        _runtime_call_db("agent_cap", ("call_cap", "ultra_media_job_create")),
    )
    session.agent_ref[0] = SimpleNamespace(interrupt=lambda reason: None)
    assert session.deadline_seconds is None
    result = json.loads(await asyncio.to_thread(
        session.invoke_platform_tool,
        "ultra_media_job_create",
        {},
        "call_cap",
    ))
    assert result["error"]["code"] == "runtime_deadline_exceeded"
    assert session.pending == {}


@pytest.mark.asyncio
async def test_interrupt_wakeup_reports_run_interrupted_not_deadline():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_intr_attr",
        asyncio.get_running_loop(),
        queue,
        [{"name": "ultra_media_job_create", "input_schema": {"type": "object"}}],
        10_000,
        "agent_intr_attr",
        _runtime_call_db("agent_intr_attr", ("call_intr", "ultra_media_job_create")),
    )
    session.agent_ref[0] = SimpleNamespace(interrupt=lambda reason: None)
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "ultra_media_job_create",
        {},
        "call_intr",
    ))
    assert (await queue.get())["type"] == "tool_request"
    session.interrupt("orchestrator stream disconnected")
    result = json.loads(await call)
    assert result["error"]["code"] == "run_interrupted"
    assert result["error"]["message"] == "run was interrupted"


@pytest.mark.asyncio
async def test_park_interrupt_defers_tool_result_without_synthetic_error():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_park",
        asyncio.get_running_loop(),
        queue,
        [{"name": "media.generate_image", "input_schema": {"type": "object"}}],
        10_000,
        "thread_park",
        _runtime_call_db("thread_park", ("call_park", "media.generate_image")),
    )
    session.agent_ref[0] = SimpleNamespace(interrupt=lambda reason: None)
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "media.generate_image",
        {},
        "call_park",
    ))
    assert (await queue.get())["type"] == "tool_request"
    session.interrupt("parked:tool_operation")

    result = await call
    assert result == DeferredToolResult("call_park")


def test_session_db_resume_appends_real_result_once_and_accepts_redelivery():
    class RecordingDB:
        def __init__(self):
            self.messages = [{
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_media",
                    "function": {"name": "media.generate_image", "arguments": "{}"},
                }],
            }]
            self.appended = []

        def append_message(self, session_id, **message):
            self.appended.append((session_id, message))
            self.messages.append({"role": message["role"], **message})

    result = {
        "tool_call_id": "call_media",
        "status": "succeeded",
        "output": {"asset_id": "asset_1", "url": "asset://image/1"},
    }
    db = RecordingDB()
    first = _resume_session_db_history(db, "thread_1", list(db.messages), [result])
    redelivered = {
        **result,
        "output": {"url": "asset://image/1", "asset_id": "asset_1"},
    }
    second = _resume_session_db_history(
        db,
        "thread_1",
        list(db.messages),
        [redelivered],
    )

    assert first[-1]["role"] == "tool"
    assert second == db.messages
    assert len(db.appended) == 1


def test_session_db_resume_survives_database_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "thread_restart"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, "api_server", model="test-model")
    db.append_message(session_id, role="user", content="make an image")
    db.append_message(
        session_id,
        role="assistant",
        content=None,
        tool_calls=[{
            "id": "call_restart",
            "type": "function",
            "function": {
                "name": "media.generate_image",
                "arguments": "{}",
            },
        }],
    )
    db.close()

    reopened = SessionDB(db_path=db_path)
    history = reopened.get_messages_as_conversation(session_id)
    result = {
        "tool_call_id": "call_restart",
        "status": "succeeded",
        "output": {"asset_id": "asset_restart"},
    }
    resumed = _resume_session_db_history(reopened, session_id, history, [result])
    assert [message["role"] for message in resumed] == ["user", "assistant", "tool"]
    reopened.close()

    verified = SessionDB(db_path=db_path)
    persisted = verified.get_messages_as_conversation(session_id)
    redelivered = _resume_session_db_history(verified, session_id, persisted, [result])
    assert redelivered == persisted
    assert [message["role"] for message in persisted] == ["user", "assistant", "tool"]
    verified.close()


@pytest.mark.asyncio
async def test_runtime_session_db_resume_rebuilds_tool_exposure_from_authoritative_history():
    class RecordingDB:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "make an image"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_previous",
                        "function": {"name": "media.generate_video", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_previous", "content": "{}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_media",
                        "function": {"name": "media.generate_image", "arguments": "{}"},
                    }],
                },
            ]

        def get_session(self, session_id):
            return {"id": session_id}

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_messages_as_conversation(self, session_id, include_ancestors=False):
            assert session_id == "thread_session"
            assert include_ancestors is True
            return list(self.messages)

        def append_message(self, session_id, **message):
            self.messages.append({"role": message["role"], **message})

    class SessionDBAdapter(_TestRuntimeAdapter):
        _api_key = ""

        def __init__(self):
            self.db = RecordingDB()

        def _check_auth(self, _request):
            return None

        def _ensure_session_db(self):
            return self.db

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                _primary_runtime={
                    "model": "configured-model",
                    "compressor_model": "configured-model",
                },
                _fallback_chain=[],
                _fallback_model=None,
                _fallback_index=0,
                _fallback_activated=False,
            )
            kwargs["agent_configurator"](agent)
            assert agent._resume_from_tool_results is True
            assert agent._require_incremental_session_persistence is True
            assert "media.generate_video" in agent.valid_tool_names
            assert "media.generate_image" in agent.valid_tool_names
            assert kwargs["session_id"] == "thread_session"
            assert kwargs["user_message"] == ""
            assert [message["role"] for message in kwargs["conversation_history"]] == [
                "user", "assistant", "tool", "assistant", "tool",
            ]
            assert kwargs["conversation_history"][-1]["tool_call_id"] == "call_media"
            return {"final_response": "done"}, {"total_tokens": 1}

    adapter = SessionDBAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/v1/runtime/runs", json=_run_body(
            "run_session_resume",
            intent="resume",
            context={"session_id": "thread_session"},
            messages=[],
            tool_results=[{
                "tool_call_id": "call_media",
                "status": "succeeded",
                "output": {"asset_id": "asset_1"},
            }],
            tools=[{
                "name": "media.generate_image",
                "description": "generate an image",
                "input_schema": {"type": "object"},
                "exposure": "deferred",
            }, {
                "name": "media.generate_video",
                "description": "generate a video",
                "input_schema": {"type": "object"},
                "exposure": "deferred",
            }],
        ))
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        assert adapter.db.messages[-1]["tool_call_id"] == "call_media"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_session_db_resume_projects_generated_output_after_durable_result():
    captured = {}

    class RecordingDB:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "make an image"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_media",
                        "function": {"name": "media.generate_image", "arguments": "{}"},
                    }],
                },
            ]

        def get_session(self, session_id):
            return {"id": session_id}

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_messages_as_conversation(self, session_id, include_ancestors=False):
            assert session_id == "thread_generated_output"
            assert include_ancestors is True
            return list(self.messages)

        def append_message(self, session_id, **message):
            self.messages.append({"role": message["role"], **message})

    class GeneratedOutputAdapter(_TestRuntimeAdapter):
        _api_key = ""

        def __init__(self):
            self.db = RecordingDB()

        def _check_auth(self, _request):
            return None

        def _ensure_session_db(self):
            return self.db

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                _primary_runtime={
                    "model": "configured-model",
                    "compressor_model": "configured-model",
                },
                _fallback_chain=[],
                _fallback_model=None,
                _fallback_index=0,
                _fallback_activated=False,
            )
            kwargs["agent_configurator"](agent)
            history = kwargs["conversation_history"]
            assert kwargs["user_message"] == ""
            assert [message["role"] for message in history] == [
                "user", "assistant", "tool",
            ]
            assert history[0]["content"] == "make an image"
            persisted_result = self.db.messages[-1]["content"]
            assert json.loads(persisted_result) == {
                "asset_id": "asset_1",
                "output_id": "output_1",
            }
            assert "runtime_generated_media_context" not in persisted_result
            projected_result = history[-1]["content"]
            assert isinstance(projected_result, str)
            assert projected_result.startswith(persisted_result)
            assert "data:image" not in projected_result
            assert "image_url=" in projected_result
            marker = projected_result
            assert "runtime_generated_media_context" not in json.dumps(history[-1])
            assert "_runtime_image_path" not in projected_result
            image_path = marker.split("image_url=", 1)[1].split(". Keep", 1)[0]
            captured["image_path"] = image_path
            assert Path(image_path).read_bytes() == b"generated-pixels"
            persisted = json.dumps(self.db.messages)
            assert image_path not in persisted
            assert "data:image" not in persisted
            assert base64.b64encode(b"generated-pixels").decode() not in persisted
            assert "runtime_generated_media_context" not in persisted
            analyzed = _runtime_tool_middleware(
                tool_name="image_analyze",
                args={"image_url": "output_1", "question": "Inspect the output"},
                session_id=kwargs["session_id"],
                tool_call_id="inspect_generated_output",
                next_call=lambda tool_args: (
                    captured.setdefault("analyze_args", tool_args)
                    and '{"success":true,"analysis":"one image"}'
                ),
            )
            assert json.loads(analyzed)["analysis"] == "one image"
            assert captured["analyze_args"]["image_url"] == image_path
            return {"final_response": "one image ready"}, {"total_tokens": 1}

    adapter = GeneratedOutputAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/v1/runtime/runs", json=_run_body(
            "run_generated_output_resume",
            intent="resume",
            context={"session_id": "thread_generated_output"},
            messages=[],
            tool_results=[{
                "tool_call_id": "call_media",
                "status": "succeeded",
                "output": {"asset_id": "asset_1", "output_id": "output_1"},
            }],
            tools=[{
                "name": "media.generate_image",
                "description": "generate an image",
                "input_schema": {"type": "object"},
            }],
            attachments=[{
                "role": "generated_output",
                "reference_id": "output_1",
                "filename": "output_1.png",
                "media_type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(b"generated-pixels").decode(),
            }],
        ))
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        assert not Path(captured["image_path"]).exists()
    finally:
        await client.close()


def test_sweeper_evicts_only_stale_finished_sessions():
    loop = asyncio.new_event_loop()
    try:
        stale = RuntimeBridgeSession("run_stale", loop, asyncio.Queue(), [], 0, "agent_stale")
        stale.finished.set()
        stale.finished_at = time.monotonic() - 10 * runtime_module._FINISHED_SESSION_TTL_SECONDS
        fresh = RuntimeBridgeSession("run_fresh", loop, asyncio.Queue(), [], 0, "agent_fresh")
        fresh.finished.set()
        fresh.finished_at = time.monotonic()
        live = RuntimeBridgeSession("run_live", loop, asyncio.Queue(), [], 0, "agent_live")
        with runtime_module._SESSIONS_LOCK:
            runtime_module._SESSIONS.update({
                "run_stale": stale,
                "run_fresh": fresh,
                "run_live": live,
            })
        removed = runtime_module._sweep_finished_sessions()
        assert removed == ["run_stale"]
        with runtime_module._SESSIONS_LOCK:
            assert "run_stale" not in runtime_module._SESSIONS
            assert "run_fresh" in runtime_module._SESSIONS
            assert "run_live" in runtime_module._SESSIONS
    finally:
        with runtime_module._SESSIONS_LOCK:
            for key in ("run_stale", "run_fresh", "run_live"):
                runtime_module._SESSIONS.pop(key, None)
        loop.close()


@pytest.mark.asyncio
async def test_runtime_run_pins_one_hour_prompt_cache_ttl():
    class TTLAdapter(_TestRuntimeAdapter):
        _api_key = ""

        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            # agent_init defaults _cache_ttl to "5m"; runtime bridge runs
            # park past that tier, so the configurator must pin "1h".
            agent = SimpleNamespace(
                tools=[], valid_tool_names=set(), model="configured-model", _cache_ttl="5m",
            )
            kwargs["agent_configurator"](agent)
            assert agent._cache_ttl == "1h"
            return {"final_response": "done"}, {"total_tokens": 1}

    adapter = TTLAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        stable = "platform rules"
        version = "ttl/v1"
        digest = "sha256:" + hashlib.sha256(
            f"{version}\nreplace\n{stable}".encode(),
        ).hexdigest()
        response = await client.post("/v1/runtime/runs", json={
            "intent": "bootstrap",
            "run_id": "run_ttl",
            "model": "chat-test",
            "context": {"session_id": "session-run-ttl"},
            "messages": [{"id": "message-ttl", "role": "user", "content": "make an image"}],
            "system_context": {
                "version": version,
                "mode": "replace",
                "stable": stable,
                "digest": digest,
            },
        })
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
        assert events[-1]["payload"]["text"] == "done"
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"failed": True, "failure_reason": "timeout"}, "provider_timeout"),
        ({"failed": True, "failure_reason": "format_error"}, "model_incompatible"),
        ({"failed": True, "failure_reason": "multimodal_tool_content_unsupported"}, "model_incompatible"),
        ({"failed": True, "turn_exit_reason": "empty_response_exhausted"}, "provider_empty_stream"),
        ({"failed": True, "error": "content_policy_blocked: rejected"}, "content_policy_blocked"),
        ({"failed": True, "error": "private downstream detail"}, "runtime_unavailable"),
    ],
)
def test_runtime_failure_code_projects_only_stable_safe_codes(result, expected):
    assert _runtime_failure_code(result) == expected


def test_runtime_persistence_failure_projects_safe_error_without_db_detail():
    result = {
        "final_response": "retained for diagnostics",
        "completed": False,
        "failed": True,
        "turn_exit_reason": "required_session_persistence_failed",
        "cleanup_errors": ["persist_session: sqlite database is locked"],
    }

    code = _runtime_failure_code(result)
    envelope = runtime_module.runtime_error_envelope(code, support_id="run-persist")

    assert code == "runtime_unavailable"
    assert envelope["code"] == "runtime_unavailable"
    assert "sqlite" not in json.dumps(envelope).lower()
    assert "database is locked" not in json.dumps(envelope).lower()


@pytest.mark.asyncio
async def test_runtime_failed_agent_result_emits_error_without_completed():
    class FailedAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
            kwargs["agent_configurator"](agent)
            return {
                "final_response": "must not be delivered as success",
                "completed": False,
                "failed": True,
                "turn_exit_reason": "empty_response_exhausted",
                "error": "private provider response",
            }, {"total_tokens": 3}

    adapter = FailedAdapter()
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/v1/runtime/runs", json=_run_body("run_failed_result"))
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert [event["type"] for event in events][-2:] == ["usage", "error"]
        assert not any(event["type"] == "completed" for event in events)
        assert events[-1]["payload"] == {
            "code": "provider_empty_stream",
            "message": "The creation service returned no output.",
            "retryable": True,
            "reason": "provider_empty_stream",
            "source": "runtime",
            "support_id": "run_failed_result",
        }
        assert "private provider response" not in json.dumps(events)
    finally:
        await client.close()


def _audit_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "gateway.api_server.audit"
    ]


class _AuditRunAdapter(_TestRuntimeAdapter):
    def _check_auth(self, _request):
        return None

    async def _run_agent_bridge(self, **kwargs):
        agent = SimpleNamespace(tools=[], valid_tool_names=set(), model="configured-model")
        kwargs["agent_configurator"](agent)
        return {"final_response": "done"}, {"total_tokens": 1}


def _audited_app(adapter: APIServerRuntimeMixin) -> web.Application:
    app = web.Application(middlewares=[request_audit_middleware])
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    app.router.add_post("/v1/runtime/runs/{run_id}/tool-results", adapter._handle_runtime_tool_result)
    app.router.add_post("/v1/runtime/runs/{run_id}/interrupt", adapter._handle_runtime_interrupt)
    return app


@pytest.mark.asyncio
async def test_runtime_run_audit_completion_line_carries_run_id(caplog):
    caplog.set_level(logging.INFO, logger="gateway.api_server.audit")
    client = TestClient(TestServer(_audited_app(_AuditRunAdapter())))
    await client.start_server()
    try:
        response = await client.post("/v1/runtime/runs", json=_run_body("run_audit_run"))
        assert response.status == 200
        events = [json.loads(line) async for line in response.content]
        assert events[-1]["type"] == "completed"
    finally:
        await client.close()

    completion_lines = [
        line for line in _audit_messages(caplog)
        if "'action': 'api.request'" in line and "'result': 'completed'" in line
    ]
    assert completion_lines, _audit_messages(caplog)
    assert "'run_id': 'run_audit_run'" in completion_lines[-1]
    # The entry line is logged before the body is parsed; run attribution
    # lives on the completion line only.
    started_lines = [
        line for line in _audit_messages(caplog) if "'result': 'started'" in line
    ]
    assert started_lines
    assert "'run_id'" not in started_lines[0]


@pytest.mark.asyncio
async def test_runtime_tool_result_audit_line_carries_run_id(caplog):
    caplog.set_level(logging.INFO, logger="gateway.api_server.audit")
    client = TestClient(TestServer(_audited_app(_AuditRunAdapter())))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/runtime/runs/run_audit_tool/tool-results",
            json={"call_id": "call_x", "ok": True, "result": {}},
        )
        # The run is not active: the handler still stamps the path run_id
        # onto the request before the session lookup, so even the 404 is
        # attributable in the audit trail.
        assert response.status == 404
    finally:
        await client.close()

    lines = [
        line for line in _audit_messages(caplog)
        if "'action': 'api.request'" in line and "'status': 404" in line
    ]
    assert lines, _audit_messages(caplog)
    assert "'run_id': 'run_audit_tool'" in lines[-1]


@pytest.mark.asyncio
async def test_runtime_interrupt_audit_line_carries_run_id(caplog):
    caplog.set_level(logging.INFO, logger="gateway.api_server.audit")

    class InterruptAdapter(_TestRuntimeAdapter):
        def _check_auth(self, _request):
            return None

        async def _run_agent_bridge(self, **kwargs):
            stop = asyncio.Event()
            agent = SimpleNamespace(
                tools=[],
                valid_tool_names=set(),
                model="configured-model",
                interrupt=lambda reason: stop.set(),
            )
            kwargs["agent_configurator"](agent)
            await stop.wait()
            return {"final_response": "stopped"}, {}

    client = TestClient(TestServer(_audited_app(InterruptAdapter())))
    await client.start_server()
    try:
        run_response = await client.post("/v1/runtime/runs", json=_run_body("run_audit_intr"))
        assert run_response.status == 200
        assert json.loads(await run_response.content.readline())["type"] == "run_started"

        interrupt_response = await client.post(
            "/v1/runtime/runs/run_audit_intr/interrupt",
            json={"reason": "audit test"},
        )
        assert interrupt_response.status == 204

        events = [json.loads(line) async for line in run_response.content]
        assert events[-1]["type"] == "completed"
    finally:
        await client.close()

    interrupt_lines = [
        line for line in _audit_messages(caplog)
        if "/v1/runtime/runs/run_audit_intr/interrupt" in line
        and "'result': 'completed'" in line
    ]
    assert interrupt_lines, _audit_messages(caplog)
    assert "'run_id': 'run_audit_intr'" in interrupt_lines[-1]
