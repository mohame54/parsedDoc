import json

from labeller import gemini


def test_request_line_shape():
    line = gemini.build_request_line("k", [gemini.text_part("hi"), gemini.file_part("files/x", "image/png")],
                                     {"type": "object"}, 100)
    assert line["key"] == "k"
    config = line["request"]["generation_config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == {"type": "object"}
    assert line["request"]["contents"][0]["parts"][1] == {
        "file_data": {"file_uri": "files/x", "mime_type": "image/png"}
    }


def test_parse_output_handles_success_and_error_lines():
    ok = {"key": "a", "response": {"candidates": [{"content": {"parts": [{"text": "{\"x\": 1}"}]}}]}}
    err = {"key": "b", "error": {"code": 429, "message": "quota"}}
    empty = {"key": "c", "response": {"candidates": []}}
    text = "\n".join(json.dumps(x) for x in (ok, err, empty)) + "\n"
    results = gemini.parse_output_text(text)
    assert results["a"] == ('{"x": 1}', None)
    assert results["b"][0] is None and "quota" in results["b"][1]
    assert results["c"][0] is None and "no candidates" in results["c"][1]


def test_prepare_image_passes_supported_formats_and_converts_others(tmp_path):
    from PIL import Image

    png = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(png)
    assert gemini.prepare_image(png, tmp_path / "s") == (png, "image/png")

    tif = tmp_path / "b.tif"
    Image.new("RGB", (4, 4)).save(tif)
    out, mime = gemini.prepare_image(tif, tmp_path / "s")
    assert out.suffix == ".png" and mime == "image/png" and out.exists()
