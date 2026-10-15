# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install langfuse
hermes plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # then check Langfuse for a "Hermes turn" trace
```

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
hermes plugins disable observability/langfuse
```

## Ultra Studio Runtime correlation

The authenticated Runtime Driver passes the product `run_id` into the agent
worker's observer context. The context is reset in `finally`, so pooled threads
cannot attach the next request to the previous run. It is telemetry only and
does not grant access or change tool execution.

All Runtime turns/resumes for one product run use a deterministic trace ID
derived from `run:<run_id>`; different runs have separate traces. The root and
child observations carry `run_id` metadata, while the Hermes session ID groups
the conversation. Ordinary CLI turns remain separate traces within a session.

API errors close their generation observation with ERROR; a later corrected
attempt remains in the same run. Tool error, blocked and cancelled outcomes
are marked explicitly. Session-end hooks close incomplete/interrupted spans.
Displayed model costs are estimates, not the platform billing ledger.

Deployment must explicitly enable this plugin and supply project credentials.
The optional `langfuse` package extra is included in the container image; merely
installing it does not enable telemetry. Configure only a dedicated dev project
for dev traffic. Payload export uses the existing secret redactor and bounds
large values; this trace is not a complete execution checkpoint.

The image pins SDK 4.7.0, which supports Langfuse v4 real-time ingestion while
remaining compatible with the existing Mistral extra's OpenTelemetry 1.39.1
constraint. Do not replace this with an unbounded SDK upgrade.
