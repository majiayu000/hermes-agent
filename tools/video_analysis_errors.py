"""Structured classification for native video-analysis failures."""

from __future__ import annotations


def classify_video_analysis_error(error: Exception) -> tuple[str, bool, str]:
    message = str(error)
    normalized = message.lower()
    if any(hint in normalized for hint in (
        "402", "insufficient", "payment required", "credits", "billing",
    )):
        return (
            "video_analysis_insufficient_credits",
            False,
            "Insufficient credits or payment required. Please top up your API "
            f"provider account and try again. Error: {message}",
        )
    if any(hint in normalized for hint in (
        "does not support", "not support video", "content_policy", "multimodal",
        "unrecognized request argument", "video input", "video_url",
    )):
        return (
            "video_analysis_model_incompatible",
            False,
            "The model does not support video analysis or the request was rejected. "
            "Ensure you are using a video-capable model "
            f"(e.g. google/gemini-2.5-flash). Error: {message}",
        )
    if any(hint in normalized for hint in (
        "too large", "payload", "413", "content_too_large",
        "request_too_large", "exceeds", "size limit",
    )):
        return (
            "video_analysis_input_too_large",
            False,
            "The video is too large for the API. Try compressing or trimming "
            f"the video (max ~50 MB). Error: {message}",
        )
    retryable = any(hint in normalized for hint in (
        "timeout", "timed out", "connection", "temporarily unavailable",
        "rate limit", "429", "500", "502", "503", "504",
    ))
    return (
        "video_analysis_failed",
        retryable,
        "There was a problem with the request and the video could not be "
        f"analyzed. Error: {message}",
    )
