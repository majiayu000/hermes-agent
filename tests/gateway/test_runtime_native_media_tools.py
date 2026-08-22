from gateway.runtime_native_media_tools import project_runtime_media_tool


def test_runtime_image_contract_preserves_name_and_accepts_all_stable_references():
    projected = project_runtime_media_tool({
        "type": "function",
        "function": {
            "name": "image_analyze",
            "description": "legacy",
            "parameters": {"properties": {
                "image_url": {"type": "string", "description": "legacy"},
                "image_paths": {"type": "string", "description": "legacy"},
            }},
        },
    })

    function = projected["function"]
    assert function["name"] == "image_analyze"
    for token in ("asset_id", "output_id", "HTTPS URL"):
        assert token in function["description"]
        assert token in function["parameters"]["properties"]["image_url"]["description"]


def test_runtime_video_contract_preserves_name_and_accepts_all_stable_references():
    projected = project_runtime_media_tool({
        "type": "function",
        "function": {
            "name": "video_analyze",
            "description": "legacy",
            "parameters": {"properties": {
                "video_url": {"type": "string", "description": "legacy"},
            }},
        },
    })

    function = projected["function"]
    assert function["name"] == "video_analyze"
    for token in ("asset_id", "output_id", "HTTPS URL"):
        assert token in function["description"]
        assert token in function["parameters"]["properties"]["video_url"]["description"]
