"""Project native context maintenance as activity, without controlling its policy."""
from __future__ import annotations

from uuid import uuid4


def bind_context_activity(agent, session) -> None:
    def observed_compression(compress):
        call_id = "context_" + uuid4().hex
        before = getattr(agent.context_compressor, "compression_count", 0)
        session.emit("activity_started", {"call_id": call_id, "name": "context_compaction", "arguments": {}})
        try:
            result = compress()
        except Exception:
            session.emit("activity_completed", {"call_id": call_id, "name": "context_compaction", "status": "failed"})
            raise
        after = getattr(agent.context_compressor, "compression_count", before)
        session.emit("activity_completed", {
            "call_id": call_id, "name": "context_compaction", "status": "completed",
            "output": {"compressed": after > before, "compression_count": after},
        })
        return result

    agent._runtime_compression_observer = observed_compression
