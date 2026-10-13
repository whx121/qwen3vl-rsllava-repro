# Qwen3-VL-4B 遥感视觉语言模型微调

使用 `Qwen/Qwen3-VL-4B-Instruct` 和 RS-LLaVA 遥感视觉指令数据，在单张 RTX 4090（24 GB）上进行 LoRA SFT。项目包含数据转换、图像检查、训练、推理和评估脚本。

本项目不是从零预训练，也不是对 RS-LLaVA 论文的严格复现。视觉编码器和视觉 Merger 保持冻结，只在 Qwen3-VL 语言模型注意力层上训练 LoRA。

## 当前实验

| 项目 | 配置 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3-VL-4B-Instruct` |
| 微调方法 | LoRA SFT |
| 数据 | RS-LLaVA 指令标注中的 RSICD 子集 |
| 训练/验证 | 7,861 / 873 张图像 |
| 训练轮数 / 步数 | 1 / 983 |
| 平均训练 loss | 0.8323 |
| GPU | NVIDIA RTX 4090 24 GB |

固定前 100 张 held-out 验证图像、603 个多轮问题的对比结果：

| 模型 | Token-F1 | ROUGE-L |
| --- | ---: | ---: |
| 原始 Qwen3-VL-4B | 0.3340 | 0.2775 |
| RSICD LoRA | 0.6110 | 0.5598 |

结果写入 `runs/qwen3vl4b_rsllava_base_eval_100/` 和 `runs/qwen3vl4b_rsllava_sft_eval_100/`。完整 873 张验证图像的命令仍保留在下文，但未在本次快速汇总中运行。

在这组相同样本上，LoRA 使 overall Token-F1 提升 `0.2770`，ROUGE-L 提升 `0.2822`，标准化 Exact Match 提升 `0.1078`。这说明当前微调确实改善了 RSICD 指令任务的回答匹配度，但仍不能据此断言模型获得了全面的遥感视觉能力。

### 50 张图像的历史上下文消融

为检查当前 LoRA 是否利用了多轮对话中的真实 reference answer，固定随机种子 `42` 抽取相同的 50 张验证图像，共 297 个问题，分别进行两种评估：

| 上下文协议 | Exact Match | Token-F1 | ROUGE-L |
| --- | ---: | ---: | ---: |
| 无历史：图片 + 当前问题 | 0.0067 | 0.4333 | 0.3819 |
| Oracle history：加入此前 reference answer | 0.1111 | 0.6097 | 0.5594 |

Oracle history 比无历史高 `0.1764` Token-F1 和 `0.1775` ROUGE-L。这个差距说明当前多轮结果不能直接视为独立单轮遥感 VQA 能力；后续应增加单轮展开训练作为对照。两组完整预测、抽样清单和运行配置分别保存在 `runs/qwen3vl4b_rsllava_sft_eval_50_no_history/` 和 `runs/qwen3vl4b_rsllava_sft_eval_50_oracle_history/`，汇总见 `results/metrics_50_history_comparison.json`。

## 基础模型和数据

基础模型：`Qwen/Qwen3-VL-4B-Instruct`

- <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- <https://github.com/QwenLM/Qwen3-VL>

训练标注和图像：

- [RS-instructions Dataset](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset)
- [原始指令 JSON](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/resolve/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json?download=true)
- [RSICD 图像](https://huggingface.co/datasets/arampacha/rsicd)
- [RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA)

本次只使用 RSICD。原始标注 39,398 条，筛选并检查本地图像后得到 8,734 条图像级样本：训练 7,861 张、验证 873 张，缺失图像为 0，训练/验证图像无重叠，随机种子为 42。当前没有官方独立 test split，10% validation split 作为 held-out 测试使用。原始图片不随代码分发。

## 微调配置

训练使用 bf16、batch size 1、梯度累积 8、1 个 epoch、learning rate `1e-5`、`model_max_length=2048`、`min_pixels=224*224`、`max_pixels=336*336` 和 gradient checkpointing。LoRA 参数为 `rank=16`、`alpha=32`、`dropout=0.05`，目标模块为 `q_proj,k_proj,v_proj,o_proj`。

LoRA 只注入语言模型 Attention 的四类投影层；视觉编码器、视觉 Merger、语言模型原始权重和 `lm_head` 均未更新。推理时需要同时加载基础模型和 adapter。

## 目录

```text
data/                     数据说明、标注和转换后的 JSON
scripts/                  数据、训练、推理和评估脚本
src/rsllava_repro/        数据处理库
tests/                    单元测试
Qwen3VL/                  本地模型和官方框架
runs/                     checkpoint 和评估输出
```

## 安装

```bash
conda activate transformer
cd /path/to/qwen3vl_rsllava_repro
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

官方训练框架：

```bash
git clone https://github.com/QwenLM/Qwen3-VL.git third_party/Qwen3-VL
export QWEN_VL_FINETUNE_ROOT="$PWD/third_party/Qwen3-VL/qwen-vl-finetune"
```

## 数据准备

将指令 JSON 放到 `data/annotations/`，将 RSICD 图片解压到 `data/raw/RSICD/images/`：

```bash
python scripts/prepare_data.py --annotation data/annotations/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json --include-prefix RSICD --path-map RSICD="$PWD/data/raw/RSICD" --out-dir data/rsllava
```

检查数据：

```bash
python scripts/validate_data.py --train data/rsllava/train.json --validation data/rsllava/validation.json
```

下载链接和目录映射见 [`data/README.md`](data/README.md)。

## 训练和推理

10 step 烟雾测试：

```bash
bash scripts/run_4090_smoke.sh --train-json data/rsllava/train.json --qwen-finetune-root "$QWEN_VL_FINETUNE_ROOT" --model-name-or-path /path/to/Qwen3VL
```

完整训练：

```bash
bash scripts/run_4090_train.sh --train-json data/rsllava/train.json --output-dir runs/qwen3vl4b_rsllava_sft --qwen-finetune-root "$QWEN_VL_FINETUNE_ROOT" --model-name-or-path /path/to/Qwen3VL
```

单图推理：

```bash
python scripts/infer.py --base-model /path/to/Qwen3VL --adapter runs/qwen3vl4b_rsllava_sft --image data/raw/RSICD/images/airport_1.jpg --prompt "Describe the remote sensing image in detail."
```

去掉 `--adapter` 即可测试基础模型。显存不足时依次降低 `--max-pixels`、`--model-max-length`，再提高梯度累积；保持视觉编码器和 Merger 冻结。

## 评估

先评估 10 个图像级样本：

```bash
python scripts/evaluate.py --model /path/to/Qwen3VL --data data/rsllava/validation.json --output runs/base_eval --max-samples 10 --max-new-tokens 64
python scripts/evaluate.py --model /path/to/Qwen3VL --adapter runs/qwen3vl4b_rsllava_sft --data data/rsllava/validation.json --output runs/sft_eval --max-samples 10 --max-new-tokens 64
```

完整 held-out 评估时去掉 `--max-samples 10`，并为两个模型使用不同的 `--output` 目录。结果包括 `metrics.json`、`predictions.jsonl`、`samples.json` 和 `errors.jsonl`。

评估脚本默认保留多轮对话历史。若要让每一个问题独立评估，使用 `--history-mode none`；若只随机抽取固定数量的图像，增加 `--random-select` 和 `--sample-seed`：

```bash
python scripts/evaluate.py \
  --model /path/to/Qwen3VL \
  --adapter runs/qwen3vl4b_rsllava_sft \
  --data data/rsllava/validation.json \
  --output runs/sft_eval_50_no_history \
  --max-samples 50 \
  --random-select \
  --sample-seed 42 \
  --history-mode none
```

`Exact Match` 是标准化后的完全匹配，`Token-F1` 是词级重合度，`ROUGE-L` 是最长公共子序列重合度。自动指标不能识别所有同义词，也不能替代人工检查。

## 测试和 GitHub 提交

```bash
pytest -q
python scripts/smoke_model.py --model /path/to/Qwen3VL --data data/rsllava/train.json
```

基础模型权重、原始图片、训练 checkpoint 和评估输出体积较大，且部分数据受上游许可证约束，不应直接提交到 GitHub。`.gitignore` 已排除这些本地文件；公开仓库保留代码、配置、数据格式说明和小型统计结果即可。

当前只使用 RSICD；不训练视觉编码器、不训练视觉 Merger、不做 DPO，也不包含从零预训练。Qwen3-VL 官方代码、模型和数据集分别遵循各自的许可证和使用条款。
