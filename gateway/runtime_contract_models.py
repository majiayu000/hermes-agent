"""Strict typed decoders for the canonical Ultra Runtime v1 contract."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)


NonEmpty128 = Annotated[str, StringConstraints(min_length=1, max_length=128)]
NonEmpty255 = Annotated[str, StringConstraints(min_length=1, max_length=255)]
NonEmpty256 = Annotated[str, StringConstraints(min_length=1, max_length=256)]
NonEmpty512 = Annotated[str, StringConstraints(min_length=1, max_length=512)]
SkillAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
ErrorCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._]*$"),
]
SHA256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
EgressGrant = Annotated[
    str,
    StringConstraints(pattern=r"^ueg_[A-Za-z0-9_-]{43}$"),
]
JsonObject: TypeAlias = dict[str, JsonValue]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RuntimeError(_StrictModel):
    code: ErrorCode
    message: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    retryable: bool
    reason: ErrorCode | None = None
    source: Literal["runtime", "orchestrator", "tool", "platform"] | None = None
    support_id: NonEmpty512 | None = None


class RuntimeMessage(_StrictModel):
    id: NonEmpty512
    role: Literal["user", "assistant", "tool"]
    content: JsonValue


class RuntimeSystemContext(_StrictModel):
    version: Annotated[str, StringConstraints(min_length=1)]
    mode: Literal["profile", "replace"]
    stable: str | None = None
    digest: SHA256Digest | None = None

    @model_validator(mode="after")
    def require_mode_fields(self) -> "RuntimeSystemContext":
        if self.mode == "replace" and (self.stable is None or self.digest is None):
            raise ValueError("replacement system context requires stable and digest")
        if self.mode == "profile" and (self.stable is not None or self.digest is not None):
            raise ValueError("profile system context accepts only version and mode")
        return self


class RuntimeContext(_StrictModel):
    session_id: NonEmpty512


class RuntimeRetryContext(_StrictModel):
    attempt: Annotated[int, Field(ge=2)]
    previous_error_code: NonEmpty128


class RuntimeRecoveryToolCall(_StrictModel):
    tool_call_id: Annotated[str, StringConstraints(min_length=1)]
    tool_name: Annotated[str, StringConstraints(min_length=1)]
    args: JsonObject


class RuntimeLLMEgress(_StrictModel):
    base_url: str
    grant: EgressGrant
    expires_at: str

    @field_validator("base_url")
    @classmethod
    def require_uri(cls, value: str) -> str:
        TypeAdapter(AnyUrl).validate_python(value)
        return value

    @field_validator("expires_at")
    @classmethod
    def require_aware_datetime(cls, value: str) -> str:
        TypeAdapter(AwareDatetime).validate_python(value)
        return value


class RuntimeAuxiliaryLLMEgress(RuntimeLLMEgress):
    model: NonEmpty256


class RuntimeRunRequest(_StrictModel):
    run_id: NonEmpty512
    model: NonEmpty512
    intent: Literal["bootstrap", "new_turn", "resume", "retry", "rebootstrap"]
    messages: list[RuntimeMessage]
    tools: list[JsonObject]
    system_context: RuntimeSystemContext
    context: RuntimeContext
    tool_results: list[JsonObject] | None = None
    runtime_context: JsonObject | None = None
    artifact_manifest: list[JsonObject] | None = None
    attachment_references: (
        dict[str, list[Annotated[str, StringConstraints(min_length=1)]]] | None
    ) = None
    skill_manifest: JsonObject | None = None
    invoked_skills: Annotated[list[SkillAlias], Field(max_length=8)] | None = None
    run_state: JsonObject | None = None
    deadline_ms: Annotated[int, Field(ge=0)] | None = None
    llm_egress: RuntimeLLMEgress | None = None
    vision_llm_egress: RuntimeAuxiliaryLLMEgress | None = None
    retry_context: RuntimeRetryContext | None = None
    recovery_tool_calls: list[RuntimeRecoveryToolCall] | None = None


class RuntimeToolSkill(_StrictModel):
    name: NonEmpty255
    digest: SHA256Digest


class RuntimeToolRequest(_StrictModel):
    call_id: NonEmpty512
    name: NonEmpty255
    arguments: JsonObject
    skills: Annotated[list[RuntimeToolSkill], Field(max_length=128)]


class RuntimeToolResult(_StrictModel):
    call_id: NonEmpty512
    ok: bool
    result: JsonValue | None = None
    error: RuntimeError | None = None

    @model_validator(mode="after")
    def require_failed_error(self) -> "RuntimeToolResult":
        if not self.ok and self.error is None:
            raise ValueError("failed tool result requires error")
        return self


class RuntimeManifestContract(_StrictModel):
    major: Literal[1]
    min_minor: Annotated[int, Field(ge=0)]
    max_minor: Annotated[int, Field(ge=0)]
    schema_digests: Annotated[list[SHA256Digest], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_digests(self) -> "RuntimeManifestContract":
        if len(self.schema_digests) != len(set(self.schema_digests)):
            raise ValueError("schema digests must be unique")
        return self


class RuntimeManifestLimits(_StrictModel):
    max_request_bytes: Annotated[int, Field(gt=0)]
    max_tool_result_bytes: Annotated[int, Field(gt=0)]


class RuntimeManifest(_StrictModel):
    runtime: Literal["hermes"]
    runtime_build: Annotated[str, StringConstraints(pattern=r"^git:[0-9a-f]{40}$")]
    contract: RuntimeManifestContract
    intents: Annotated[
        list[Literal["bootstrap", "new_turn", "resume", "retry", "rebootstrap"]],
        Field(min_length=1),
    ]
    features: Annotated[
        list[
            Annotated[
                str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]+\.v[1-9][0-9]*$")
            ]
        ],
        Field(min_length=1),
    ]
    limits: RuntimeManifestLimits

    @model_validator(mode="after")
    def require_unique_capabilities(self) -> "RuntimeManifest":
        if len(self.intents) != len(set(self.intents)):
            raise ValueError("manifest intents must be unique")
        if len(self.features) != len(set(self.features)):
            raise ValueError("manifest features must be unique")
        return self


class RunStartedPayload(_StrictModel):
    runtime: Literal["hermes"]
    system_context_version: Annotated[str, StringConstraints(min_length=1)]
    system_context_mode: Literal["profile", "replace"]
    system_context_digest: SHA256Digest


class EmptyPayload(_StrictModel):
    pass


class TextDeltaPayload(_StrictModel):
    delta: str


class RuntimeModelContractControlRequest(_StrictModel):
    request_id: NonEmpty128
    kind: Literal["model_contract.get"]
    model: NonEmpty512


class RuntimeMediaReferenceControlRequest(_StrictModel):
    request_id: NonEmpty128
    kind: Literal["media_reference.resolve"]
    reference_id: NonEmpty512
    media_type: Literal["image"]


class RuntimeVideoEvidenceControlRequest(_StrictModel):
    request_id: NonEmpty128
    kind: Literal["video_evidence.prepare"]
    reference_id: NonEmpty512
    media_type: Literal["video"]
    include_transcript: bool


RuntimeControlRequestPayload: TypeAlias = (
    RuntimeModelContractControlRequest
    | RuntimeMediaReferenceControlRequest
    | RuntimeVideoEvidenceControlRequest
)


class ActivityStartedPayload(_StrictModel):
    call_id: NonEmpty512
    name: NonEmpty255
    arguments: JsonObject


class ActivityCompletedPayload(_StrictModel):
    call_id: NonEmpty512
    name: NonEmpty255
    status: Literal["completed", "failed"]
    arguments: JsonObject | None = None
    error: RuntimeError | None = None


class CompletedPayload(_StrictModel):
    finish_reason: Literal["stop"]
    text: str


class _RunStartedEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["run_started"]
    payload: RunStartedPayload


class _HeartbeatEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["heartbeat"]
    payload: EmptyPayload


class _TextDeltaEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["text_delta"]
    payload: TextDeltaPayload


class _ControlRequestEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["runtime_control_request"]
    payload: RuntimeControlRequestPayload


class _ToolRequestEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["tool_request"]
    payload: RuntimeToolRequest


class _ActivityStartedEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["activity_started"]
    payload: ActivityStartedPayload


class _ActivityCompletedEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["activity_completed"]
    payload: ActivityCompletedPayload


class _UsageEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["usage"]
    payload: JsonObject


class _CompletedEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["completed"]
    payload: CompletedPayload


class _ErrorEvent(_StrictModel):
    run_id: NonEmpty512 | None = None
    type: Literal["error"]
    payload: RuntimeError


RuntimeEvent: TypeAlias = Annotated[
    _RunStartedEvent
    | _HeartbeatEvent
    | _TextDeltaEvent
    | _ControlRequestEvent
    | _ToolRequestEvent
    | _ActivityStartedEvent
    | _ActivityCompletedEvent
    | _UsageEvent
    | _CompletedEvent
    | _ErrorEvent,
    Field(discriminator="type"),
]
_RUNTIME_EVENT_ADAPTER = TypeAdapter(RuntimeEvent)


def decode_runtime_run_request(value: object) -> RuntimeRunRequest:
    return RuntimeRunRequest.model_validate(value)


def decode_runtime_tool_result(value: object) -> RuntimeToolResult:
    return RuntimeToolResult.model_validate(value)


def decode_runtime_tool_request(value: object) -> RuntimeToolRequest:
    return RuntimeToolRequest.model_validate(value)


def decode_runtime_manifest(value: object) -> RuntimeManifest:
    return RuntimeManifest.model_validate(value)


def decode_runtime_error(value: object) -> RuntimeError:
    return RuntimeError.model_validate(value)


def decode_runtime_event(value: object) -> RuntimeEvent:
    return _RUNTIME_EVENT_ADAPTER.validate_python(value)


def encode_runtime_event(value: object) -> bytes:
    """Encode one Runtime event only after the strict consumer accepts it."""
    event = decode_runtime_event(value)
    return json.dumps(
        event.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
