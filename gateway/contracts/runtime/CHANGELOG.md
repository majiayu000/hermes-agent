# Runtime contract changelog

## 1.8.0 — 2026-10-20

Require `execution_subscription.v1`. Start, attach, events and controls bind
`execution_id` and `lease_epoch` query parameters in the signed capability.
Start is idempotent for the same execution and canonical request digest. A
higher lease epoch fences older subscriptions and control requests. Responses
identify the execution with `X-Runtime-Execution-Id`.

`GET /v1/runtime/runs/{run_id}/events` replays the original NDJSON journal.
`POST /v1/runtime/runs/{run_id}/attach` also binds the new host's same-model,
same-route LLM capabilities (`model`, `llm_egress`, `vision_llm_egress`). It does
not restart the Agent or extend the original deadline. An active execution
missing after a Runtime process restart returns `runtime_execution_not_found`.

The host persists a cursor over normalized product events. It replays transport
frames to rebuild decoder state and skips already consumed product events.
Source-key event receipts and an unacknowledged tool-result envelope close the
commit/delivery crash windows; existing tool ledger identities own side effects.
No separate ACK endpoint or mixed-version compatibility path is introduced.
Upgrade both sides together after draining affected active Runs.

## 1.7.0 — 2026-10-18

Resume tool_results use the same call_id/ok/result/error envelope as live tool
results. Preserve independent error and result evidence and structured output
references. Upgrade producer and consumer together; the old resume envelope is
not accepted. No executor or provider operation is replayed by this change.

## 1.6.0 — 2026-09-10

Tool errors require code/message and preserve business extensions without a retry
verdict requirement or silent text truncation. Failure result bodies and complete
output references cross the same boundary as successful results. Product lifecycle
errors retain their separate strict schema. Upgrade producer and consumer together.


## 1.5.0 — 2026-08-24

- Remove inline source-media `attachments` from Run requests; initial and
  resumed Runs carry opaque `attachment_references` only.
- Restrict `media_reference.resolve` to bounded images and add
  `video_evidence.prepare` for digest-addressed uniform JPEG samples plus an
  optional bounded audio proxy.

## 1.4.0 — 2026-08-23

- Replace the untyped Runtime interrupt control with negotiated suspend,
  cancel, and abort controls. Suspend preserves the Hermes continuation and
  never invokes the Agent interrupt API.

## 1.3.0 — 2026-08-22

- Add private on-demand `media_reference.resolve` control requests for run-bound
  `asset_id` and `output_id` references used by Runtime-native media analysis.
- Keep resolved bytes and paths ephemeral and outside durable product events.

## 1.2.0 — 2026-08-19

- Add Hermes-owned named system-prompt profiles.
- Keep legacy replacement artifacts readable for recovery of existing Runs.

## 1.1.0 — 2026-08-19

- Add structured `invoked_skills` to the private Run request.
- Keep Skill activation intent out of Orchestrator-authored system-prompt prose.

## 1.0.0 — 2026-08-13

- Establish the Agent Orchestrator as the canonical Runtime contract owner.
- Define strict schemas for manifest, run request, NDJSON event, tool request,
  tool result, and safe error envelopes.
- Advertise one deterministic bundle digest across every schema in v1.
- Cover the current Hermes capabilities for LLM egress, vision egress,
  tool-result replay, and SessionDB rebootstrap.
