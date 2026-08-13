# Runtime contract changelog

## 1.0.0 — 2026-08-13

- Establish the Agent Orchestrator as the canonical Runtime contract owner.
- Define strict schemas for manifest, run request, NDJSON event, tool request,
  tool result, and safe error envelopes.
- Advertise one deterministic bundle digest across every schema in v1.
- Cover the current Hermes capabilities for LLM egress, vision egress,
  tool-result replay, and SessionDB rebootstrap.
