from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

IMAGE_TOKEN = "<image>"
_IMAGE_RE = re.compile(r"<image>\s*", flags=re.IGNORECASE)


class DataFormatError(ValueError):
    """Raised when a row cannot be converted to the Qwen-VL format."""


@dataclass
class Rejection:
    index: int
    reason: str
    image: str | None = None


@dataclass
class DataStats:
    input_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    unique_images: int = 0
    total_turns: int = 0
    caption_rows: int = 0
    vqa_rows: int = 0
    multi_turn_rows: int = 0
    missing_images: int = 0
    duplicate_image_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataFormatError(f"{path}:{line_number} is not an object")
                rows.append(value)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "annotations", "train", "validation"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)]
    raise DataFormatError(f"Cannot find a list of records in {path}")


def load_huggingface_rows(dataset_name: str, split: str = "train", config: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Install the project dependencies to load a Hugging Face dataset") from exc
    kwargs: dict[str, Any] = {"split": split}
    if config:
        dataset = load_dataset(dataset_name, config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in dataset]


def filter_rows_by_image_prefix(rows: Iterable[Mapping[str, Any]], prefixes: Iterable[str]) -> list[dict[str, Any]]:
    """Keep rows whose annotation image path starts with one dataset prefix."""
    normalized = tuple(prefix.strip().strip("/") + "/" for prefix in prefixes if prefix.strip())
    if not normalized:
        return [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    for row in rows:
        image = row.get("image", row.get("image_path")) if isinstance(row, Mapping) else None
        if isinstance(image, str) and image.lstrip("/").startswith(normalized):
            selected.append(dict(row))
    return selected


def parse_path_maps(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-map must be PREFIX=PATH, got {value!r}")
        prefix, path = value.split("=", 1)
        prefix = prefix.strip().strip("/")
        if not prefix or not path:
            raise ValueError(f"--path-map must be PREFIX=PATH, got {value!r}")
        result[prefix] = Path(path).expanduser().resolve()
    return result


def resolve_image_path(raw_image: str, data_root: Path, path_maps: Mapping[str, Path]) -> Path:
    candidate = Path(raw_image).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    parts = candidate.parts
    if parts and parts[0] in path_maps:
        return (path_maps[parts[0]] / Path(*parts[1:])).resolve()
    return (data_root / candidate).resolve()


def _raw_conversations(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    conversations = row.get("conversations", row.get("messages"))
    if not isinstance(conversations, list):
        raise DataFormatError("missing conversations/messages list")
    return [item for item in conversations if isinstance(item, Mapping)]


def _role(value: Any) -> str:
    role = str(value).strip().lower()
    aliases = {"human": "human", "user": "human", "question": "human", "gpt": "gpt", "assistant": "gpt", "answer": "gpt"}
    if role not in aliases:
        raise DataFormatError(f"unsupported conversation role: {value!r}")
    return aliases[role]


def _remove_image_tokens(text: str) -> str:
    return _IMAGE_RE.sub("", text).strip()


def normalize_conversations(raw: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not raw:
        raise DataFormatError("empty conversations")
    output: list[dict[str, str]] = []
    first_image_seen = False
    previous_role: str | None = None
    for item in raw:
        role = _role(item.get("from", item.get("role")))
        text = item.get("value", item.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise DataFormatError("conversation message has empty text")
        if role == "gpt" and _IMAGE_RE.search(text):
            raise DataFormatError("assistant answer contains <image>")
        if role == "human":
            text = _remove_image_tokens(text)
            if not first_image_seen:
                text = f"{IMAGE_TOKEN}\n{text}".strip()
                first_image_seen = True
            if not text.replace(IMAGE_TOKEN, "").strip():
                raise DataFormatError("human message has no text after removing image tokens")
        if previous_role == role:
            raise DataFormatError("conversation roles are not alternating")
        output.append({"from": role, "value": text})
        previous_role = role
    if len(output) < 2 or output[0]["from"] != "human" or output[-1]["from"] != "gpt":
        raise DataFormatError("conversation must start with human and end with gpt")
    if len(output) % 2 != 0:
        raise DataFormatError("conversation must contain complete human/gpt pairs")
    return output


def classify_question(question: str) -> str:
    value = question.lower()
    caption_terms = ("describe", "caption", "what does the image show", "what is depicted", "what kind of scene")
    return "caption" if any(term in value for term in caption_terms) else "vqa"


def normalize_row(row: Mapping[str, Any], index: int, data_root: Path, path_maps: Mapping[str, Path], allow_missing: bool = False) -> tuple[dict[str, Any], bool]:
    raw_image = row.get("image", row.get("image_path"))
    if isinstance(raw_image, list):
        if len(raw_image) != 1:
            raise DataFormatError("only single-image records are supported in v1")
        raw_image = raw_image[0]
    if not isinstance(raw_image, str) or not raw_image.strip():
        raise DataFormatError("missing image path")
    image_path = resolve_image_path(raw_image, data_root, path_maps)
    missing = not image_path.is_file()
    if missing and not allow_missing:
        raise DataFormatError(f"missing image: {image_path}")
    conversations = normalize_conversations(_raw_conversations(row))
    record_id = str(row.get("id", Path(raw_image).stem or f"row-{index}"))
    return {
        "id": record_id,
        "conversation_id": record_id,
        "image": str(image_path),
        "source_image": raw_image,
        "conversations": conversations,
        "task_type": classify_question(conversations[0]["value"].replace(IMAGE_TOKEN, "").strip()),
    }, missing


def prepare_records(rows: Iterable[Mapping[str, Any]], data_root: Path, path_maps: Mapping[str, Path], allow_missing: bool = False) -> tuple[list[dict[str, Any]], list[Rejection], DataStats]:
    records: list[dict[str, Any]] = []
    rejected: list[Rejection] = []
    stats = DataStats()
    seen_images: Counter[str] = Counter()
    rows = list(rows)
    stats.input_rows = len(rows)
    for index, row in enumerate(rows):
        raw_image = row.get("image", row.get("image_path")) if isinstance(row, Mapping) else None
        raw_image_text = raw_image if isinstance(raw_image, str) else None
        try:
            record, missing = normalize_row(row, index, data_root, path_maps, allow_missing=allow_missing)
            if missing:
                stats.missing_images += 1
            records.append(record)
            seen_images[record["image"]] += 1
            stats.total_turns += len(record["conversations"]) // 2
            stats.multi_turn_rows += int(len(record["conversations"]) > 2)
            stats.caption_rows += int(record["task_type"] == "caption")
            stats.vqa_rows += int(record["task_type"] == "vqa")
        except (DataFormatError, TypeError, KeyError) as exc:
            rejected.append(Rejection(index=index, reason=str(exc), image=raw_image_text))
    stats.accepted_rows = len(records)
    stats.rejected_rows = len(rejected)
    stats.unique_images = len(seen_images)
    stats.duplicate_image_rows = sum(count - 1 for count in seen_images.values() if count > 1)
    return records, rejected, stats


def split_by_image(records: list[dict[str, Any]], validation_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["image"], []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_count = max(1, round(len(keys) * validation_ratio)) if len(keys) > 1 else 0
    validation_keys = set(keys[:validation_count])
    validation = [record for key in keys if key in validation_keys for record in groups[key]]
    train = [record for key in keys if key not in validation_keys for record in groups[key]]
    return train, validation


def validate_split_files(train: Iterable[Mapping[str, Any]], validation: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    train = list(train)
    validation = list(validation)
    train_images = {str(row["image"]) for row in train}
    validation_images = {str(row["image"]) for row in validation}
    overlap = sorted(train_images & validation_images)
    invalid = []
    for split_name, rows in (("train", train), ("validation", validation)):
        for index, row in enumerate(rows):
            image = Path(str(row.get("image", "")))
            conversations = row.get("conversations")
            if not image.is_file():
                invalid.append({"split": split_name, "index": index, "reason": "missing_image", "image": str(image)})
            if not isinstance(conversations, list) or len(conversations) < 2:
                invalid.append({"split": split_name, "index": index, "reason": "invalid_conversations"})
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_images": len(train_images),
        "validation_images": len(validation_images),
        "image_overlap": overlap,
        "invalid_rows": invalid,
        "valid": not overlap and not invalid,
    }
