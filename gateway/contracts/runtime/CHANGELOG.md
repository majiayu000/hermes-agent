# Runtime contract changelog

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
