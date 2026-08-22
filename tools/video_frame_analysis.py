"""Bounded, duration-aware server-side frame extraction for video analysis."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path

from agent.video_post.ffmpeg import FfmpegError, probe_media, run_ffmpeg


TARGET_FRAME_INTERVAL_SECONDS = 2.5
MIN_FRAME_COUNT = 3
MAX_FRAME_COUNT = 24
_FRAME_EDGE_PX = 1280
_FRAME_TIMEOUT_SECONDS = 60.0


class VideoFrameExtractionError(RuntimeError):
    """Raised when a source cannot produce the required representative frames."""


@dataclass(frozen=True)
class ExtractedVideoFrame:
    path: Path
    timestamp_seconds: float
    ratio: float


def plan_video_frame_ratios(duration_seconds: float) -> tuple[float, ...]:
    """Plan bounded, evenly distributed timeline samples for a video duration."""
    if duration_seconds <= 0:
        raise VideoFrameExtractionError("Video duration must be greater than zero.")
    frame_count = min(
        MAX_FRAME_COUNT,
        max(MIN_FRAME_COUNT, math.ceil(duration_seconds / TARGET_FRAME_INTERVAL_SECONDS)),
    )
    return tuple((index + 0.5) / frame_count for index in range(frame_count))


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
) -> list[ExtractedVideoFrame]:
    """Extract a bounded, duration-aware set of JPEG timeline samples."""
    source = video_path.resolve()
    destination = output_dir.resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        media = probe_media(str(source))
    except FfmpegError as exc:
        raise VideoFrameExtractionError(f"Unable to inspect video: {exc}") from exc

    duration = float(media.get("duration") or 0.0)
    if not media.get("has_video") or duration <= 0:
        raise VideoFrameExtractionError(
            "Video must contain a valid video stream and duration."
        )

    frame_ratios = plan_video_frame_ratios(duration)
    frames: list[ExtractedVideoFrame] = []
    for index, ratio in enumerate(frame_ratios, start=1):
        timestamp = min(max(0.0, duration * ratio), max(0.0, duration - 0.001))
        frame_path = destination / f"frame-{index}.jpg"
        try:
            run_ffmpeg(
                [
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-vf",
                    (
                        f"scale={_FRAME_EDGE_PX}:{_FRAME_EDGE_PX}:"
                        "force_original_aspect_ratio=decrease"
                    ),
                    "-q:v",
                    "3",
                    str(frame_path),
                ],
                timeout=_FRAME_TIMEOUT_SECONDS,
            )
        except FfmpegError as exc:
            raise VideoFrameExtractionError(
                f"Unable to extract video timeline frame {index}: {exc}"
            ) from exc
        if not frame_path.is_file() or frame_path.stat().st_size <= 0:
            raise VideoFrameExtractionError(
                f"Video timeline extraction did not produce a JPEG for frame {index}."
            )
        frames.append(
            ExtractedVideoFrame(
                path=frame_path,
                timestamp_seconds=timestamp,
                ratio=ratio,
            )
        )
    return frames


def build_video_frame_analysis_message(
    user_prompt: str,
    frames: list[ExtractedVideoFrame],
) -> dict[str, object]:
    """Package timestamped JPEG samples for an OpenAI-compatible vision call."""
    content: list[dict[str, object]] = [{
        "type": "text",
        "text": (
            f"{user_prompt}\n\n"
            "The source video was sampled server-side across its timeline at the "
            "labeled timestamps. Analyze only what these frames support, preserve their "
            "chronological order, and disclose when a conclusion would require "
            "unsampled motion or timing."
        ),
    }]
    for index, frame in enumerate(frames, start=1):
        content.append({
            "type": "text",
            "text": (
                f"Timeline frame {index}/{len(frames)} at "
                f"{_format_timestamp(frame.timestamp_seconds)} "
                f"({frame.ratio:.0%} of source duration)."
            ),
        })
        encoded = base64.b64encode(frame.path.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })
    return {"role": "user", "content": content}


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
