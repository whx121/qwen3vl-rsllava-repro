from pathlib import Path

import pytest

from rsllava_repro.data import DataFormatError, filter_rows_by_image_prefix, normalize_conversations, prepare_records, split_by_image, validate_split_files


def _row(image: str = "RSICD/images/a.jpg"):
    return {
        "id": "a",
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe the image."},
            {"from": "gpt", "value": "A remote-sensing scene."},
        ],
    }


def test_normalize_multiturn_removes_repeated_image_tokens():
    result = normalize_conversations([
        {"from": "human", "value": "<image>\nDescribe it."},
        {"from": "gpt", "value": "A city."},
        {"from": "human", "value": "<image>\nHow many buildings?"},
        {"from": "gpt", "value": "Several."},
    ])
    assert result[0]["value"].startswith("<image>")
    assert "<image>" not in result[2]["value"]


def test_answer_image_token_is_rejected():
    with pytest.raises(DataFormatError):
        normalize_conversations([
            {"from": "human", "value": "<image>\nDescribe it."},
            {"from": "gpt", "value": "<image>"},
        ])


def test_prepare_records_reports_missing_images(tmp_path: Path):
    records, rejected, stats = prepare_records([_row()], tmp_path, {}, allow_missing=False)
    assert not records
    assert len(rejected) == 1
    assert stats.rejected_rows == 1


def test_split_by_image_has_no_overlap(tmp_path: Path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.touch()
    image_b.touch()
    rows = []
    for image in (image_a, image_b):
        for index in range(3):
            row = _row(str(image))
            row["id"] = f"{image.stem}-{index}"
            rows.append(row)
    train, validation = split_by_image(rows, validation_ratio=0.5, seed=42)
    report = validate_split_files(train, validation)
    assert report["image_overlap"] == []
    assert report["valid"]


def test_filter_rows_by_image_prefix():
    rows = [_row("RSICD/images/a.jpg"), _row("UCM/images/b.jpg")]
    assert len(filter_rows_by_image_prefix(rows, ["RSICD"])) == 1
