from pathlib import Path
from unittest.mock import patch

import pytest

from tools.video_frame_analysis import (
    FRAME_RATIOS,
    VideoFrameExtractionError,
    extract_video_frames,
)


def test_extract_video_frames_uses_bounded_server_side_sampling(tmp_path):
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
        return_value={"has_video": True, "duration": 100.0},
    ), patch("tools.video_frame_analysis.run_ffmpeg", side_effect=fake_run):
        frames = extract_video_frames(video_path, output_dir)

    assert FRAME_RATIOS == (0.15, 0.5, 0.85)
    assert [frame.timestamp_seconds for frame in frames] == [15.0, 50.0, 85.0]
    assert [frame.ratio for frame in frames] == list(FRAME_RATIOS)
    assert all(frame.path.read_bytes() == b"jpeg" for frame in frames)
    assert [command[command.index("-ss") + 1] for command in commands] == [
        "15.000",
        "50.000",
        "85.000",
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
