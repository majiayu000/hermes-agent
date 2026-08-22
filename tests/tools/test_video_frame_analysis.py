from pathlib import Path
from unittest.mock import patch

import pytest

from tools.video_frame_analysis import (
    MAX_FRAME_COUNT,
    MIN_FRAME_COUNT,
    TARGET_FRAME_INTERVAL_SECONDS,
    VideoFrameExtractionError,
    extract_video_frames,
    plan_video_frame_ratios,
)


@pytest.mark.parametrize(
    ("duration", "expected_count"),
    [
        (1.0, MIN_FRAME_COUNT),
        (7.5, 3),
        (10.0, 4),
        (30.0, 12),
        (60.0, MAX_FRAME_COUNT),
        (600.0, MAX_FRAME_COUNT),
    ],
)
def test_plan_video_frame_ratios_adapts_to_duration_with_a_hard_cap(
    duration,
    expected_count,
):
    ratios = plan_video_frame_ratios(duration)

    assert TARGET_FRAME_INTERVAL_SECONDS == 2.5
    assert len(ratios) == expected_count
    assert list(ratios) == sorted(ratios)
    assert all(0 < ratio < 1 for ratio in ratios)
    assert ratios[0] == pytest.approx(0.5 / expected_count)
    assert ratios[-1] == pytest.approx((expected_count - 0.5) / expected_count)


def test_plan_video_frame_ratios_rejects_non_positive_duration():
    with pytest.raises(VideoFrameExtractionError, match="greater than zero"):
        plan_video_frame_ratios(0)


def test_extract_video_frames_uses_duration_aware_server_side_sampling(tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "frames"
    commands: list[list[str]] = []

    def fake_run(args, *, timeout):
        commands.append(args)
        Path(args[-1]).write_bytes(b"jpeg")
        assert timeout == 60.0

    with patch(
        "tools.video_frame_analysis.probe_media",
        return_value={"has_video": True, "duration": 10.0},
    ), patch("tools.video_frame_analysis.run_ffmpeg", side_effect=fake_run):
        frames = extract_video_frames(video_path, output_dir)

    assert [frame.timestamp_seconds for frame in frames] == [1.25, 3.75, 6.25, 8.75]
    assert [frame.ratio for frame in frames] == [0.125, 0.375, 0.625, 0.875]
    assert all(frame.path.read_bytes() == b"jpeg" for frame in frames)
    assert [command[command.index("-ss") + 1] for command in commands] == [
        "1.250",
        "3.750",
        "6.250",
        "8.750",
    ]
    assert all("scale=1280:1280:force_original_aspect_ratio=decrease" in command for command in commands)


@pytest.mark.parametrize(
    "probe_result",
    [
        {"has_video": False, "duration": 12.0},
        {"has_video": True, "duration": 0.0},
    ],
)
def test_extract_video_frames_rejects_invalid_video_before_ffmpeg(tmp_path, probe_result):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")

    with patch("tools.video_frame_analysis.probe_media", return_value=probe_result), patch(
        "tools.video_frame_analysis.run_ffmpeg"
    ) as run_ffmpeg:
        with pytest.raises(VideoFrameExtractionError, match="valid video stream and duration"):
            extract_video_frames(video_path, tmp_path / "frames")

    run_ffmpeg.assert_not_called()


def test_extract_video_frames_rejects_empty_output(tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")

    with patch(
        "tools.video_frame_analysis.probe_media",
        return_value={"has_video": True, "duration": 10.0},
    ), patch("tools.video_frame_analysis.run_ffmpeg"):
        with pytest.raises(VideoFrameExtractionError, match="did not produce a JPEG"):
            extract_video_frames(video_path, tmp_path / "frames")
