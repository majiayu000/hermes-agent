# Runtime contract changelog

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
