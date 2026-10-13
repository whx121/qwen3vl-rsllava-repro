#!/usr/bin/env python3
"""Launch the official Qwen3-VL trainer with a temporary local data registry."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-4B on prepared RS-LLaVA data")
    parser.add_argument("--qwen-finetune-root", type=Path, default=Path(os.environ.get("QWEN_VL_FINETUNE_ROOT", "third_party/Qwen3-VL/qwen-vl-finetune")))
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=336 * 336)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--flash-attention-2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def materialize_framework(source: Path, output_dir: Path, annotation_path: Path) -> Path:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not (source / "qwenvl" / "train" / "train_qwen.py").is_file():
        raise FileNotFoundError(f"Expected official trainer at {source / 'qwenvl/train/train_qwen.py'}")
    isolated = output_dir / ".qwen-vl-finetune-runtime"
    if not isolated.exists():
        shutil.copytree(source, isolated)
    registry = isolated / "qwenvl" / "data" / "__init__.py"
    if not registry.is_file():
        raise FileNotFoundError(f"Cannot find upstream dataset registry: {registry}")
    marker = "# rsllava-repro dataset registration"
    text = registry.read_text(encoding="utf-8")
    registration = (
        f"\n{marker}\n"
        "data_dict[\"rsllava_repro\"] = {\n"
        f"    \"annotation_path\": {json.dumps(str(annotation_path.resolve()))},\n"
        "    \"data_path\": \"\",\n"
        "}\n"
    )
    # The runtime is deliberately disposable.  Replace an earlier registration
    # so reusing an output directory cannot silently train on a stale subset.
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    registry.write_text(text + registration, encoding="utf-8")
    patch_optional_flash_attention(isolated)
    return isolated


def patch_optional_flash_attention(runtime: Path) -> None:
    """Keep the disposable upstream copy usable without flash-attn.

    The upstream trainer imports flash-attn unconditionally even when packed
    data is disabled.  In the default path its custom flash function is never
    called, so an import stub plus Transformers SDPA is sufficient for a
    correctness smoke test.  The source checkout remains untouched.
    """
    trainer = runtime / "qwenvl" / "train" / "trainer.py"
    text = trainer.read_text(encoding="utf-8")
    old_import = "from flash_attn.flash_attn_interface import flash_attn_varlen_func"
    new_import = (
        "try:\n"
        "    " + old_import + "\n"
        "except ImportError:  # optional dependency; unused when data_packing=False\n"
        "    def flash_attn_varlen_func(*args, **kwargs):\n"
        "        raise RuntimeError('flash-attn is required only for packed-data attention')\n"
    )
    if old_import in text and "optional dependency; unused" not in text:
        trainer.write_text(text.replace(old_import, new_import, 1), encoding="utf-8")

    train_script = runtime / "qwenvl" / "train" / "train_qwen.py"
    text = train_script.read_text(encoding="utf-8")
    # Keep the function signature intact and change only the __main__ call.
    text = text.replace(
        'def train(attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa")):',
        'def train(attn_implementation="flash_attention_2"):',
        1,
    )
    old_main = 'if __name__ == "__main__":\n    train(attn_implementation="flash_attention_2")'
    new_main = 'if __name__ == "__main__":\n    train(attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"))'
    if old_main in text:
        text = text.replace(old_main, new_main, 1)

    old_resume = '''    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()'''
    new_resume = '''    resume_checkpoint = os.environ.get("QWEN_RESUME_FROM_CHECKPOINT")
    if resume_checkpoint:
        logging.info("resuming from explicit checkpoint %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    elif list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()'''
    if old_resume in text:
        text = text.replace(old_resume, new_resume, 1)
    train_script.write_text(text, encoding="utf-8")


def make_subset(path: Path, output_dir: Path, max_samples: int | None) -> Path:
    if max_samples is None:
        return path.resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Training file must contain a JSON list: {path}")
    subset = output_dir / "train_subset.json"
    subset.write_text(json.dumps(rows[:max_samples], ensure_ascii=False, indent=2), encoding="utf-8")
    return subset.resolve()


def build_command(args: argparse.Namespace, framework: Path, annotation: Path) -> list[str]:
    # `shutil.which("torchrun")` can resolve to a different Conda env than the
    # Python interpreter used to launch this wrapper.  Keep the whole run in
    # the selected environment, especially important for CUDA builds.
    env_torchrun = Path(sys.executable).with_name("torchrun")
    torchrun = str(env_torchrun) if env_torchrun.is_file() else None
    command = [torchrun or sys.executable]
    if not torchrun:
        command += ["-m", "torch.distributed.run"]
    command += ["--nproc_per_node=1", str(framework / "qwenvl/train/train_qwen.py")]
    pairs = {
        "--model_name_or_path": args.model_name_or_path,
        "--tune_mm_llm": "True",
        "--tune_mm_vision": "False",
        "--tune_mm_mlp": "False",
        "--dataset_use": "rsllava_repro",
        "--output_dir": str(args.output_dir.resolve()),
        "--cache_dir": str(args.cache_dir.resolve()),
        "--bf16": "True",
        "--per_device_train_batch_size": args.per_device_train_batch_size,
        "--gradient_accumulation_steps": args.gradient_accumulation_steps,
        "--learning_rate": args.learning_rate,
        "--model_max_length": args.model_max_length,
        "--max_pixels": args.max_pixels,
        "--min_pixels": args.min_pixels,
        "--num_train_epochs": args.num_train_epochs,
        "--lr_scheduler_type": "cosine",
        "--weight_decay": 0.01,
        "--logging_steps": args.logging_steps,
        "--save_steps": args.save_steps,
        "--save_total_limit": args.save_total_limit,
        "--lora_enable": "True",
        "--lora_r": args.lora_r,
        "--lora_alpha": args.lora_alpha,
        "--lora_dropout": args.lora_dropout,
        "--gradient_checkpointing": "True",
        "--data_flatten": "False",
        "--data_packing": "False",
        "--report_to": "none",
        "--remove_unused_columns": "False",
    }
    for key, value in pairs.items():
        command += [key, str(value)]
    if args.max_steps is not None:
        command += ["--max_steps", str(args.max_steps)]
    return command


def main() -> int:
    args = parse_args()
    local_model = Path(args.model_name_or_path).expanduser()
    if local_model.is_dir():
        args.model_name_or_path = str(local_model.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    annotation = make_subset(args.train_json, args.output_dir, args.max_samples)
    framework = materialize_framework(args.qwen_finetune_root, args.output_dir, annotation)
    command = build_command(args, framework, annotation)
    print("Launching:")
    print(" ".join(shlex.quote(item) for item in command))
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment["QWEN_ATTN_IMPLEMENTATION"] = "flash_attention_2" if args.flash_attention_2 else "sdpa"
    if args.resume_from_checkpoint:
        environment["QWEN_RESUME_FROM_CHECKPOINT"] = str(args.resume_from_checkpoint.expanduser().resolve())
    return subprocess.run(command, cwd=framework, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
