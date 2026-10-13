# 数据目录说明

本项目不会把遥感原图复制进代码仓库。RS-LLaVA 标注和原始图像分别来自外部数据集，使用者需要按各数据集许可自行准备。

推荐目录结构如下：

```text
data/
├── raw/
│   ├── RSICD/images/
│   ├── UCM/images/
│   ├── UAV/images/
│   ├── RSVQA-LR/images/
│   └── RSIVQA-DOTA/images/
├── annotations/
└── rsllava/
    ├── train.json
    ├── validation.json
    └── manifest.json
```

## 最容易跑通的第一版：只下载 RSICD

RSICD 已经有可直接读取图像的 Hugging Face 镜像，约 10,921 张图、526 MB：

<https://huggingface.co/datasets/arampacha/rsicd>

在项目目录执行：

```bash
conda activate transformer
python scripts/download_rsicd.py --out-dir data/raw/RSICD
```

如果你从 Hugging Face 网页下载到的是 `train-*.parquet`、`test-*.parquet`
和 `valid-*.parquet`，它们内部已经包含图片二进制，但还没有展开成 jpg。
在本项目当前目录结构下执行：

```bash
python scripts/extract_rsicd_parquet.py \
  --parquet-dir data/raw/RSICD/images \
  --out-dir data/raw/RSICD
```

这会生成：

```text
data/raw/RSICD/images/airport_1.jpg
data/raw/RSICD/images/...
```

然后只使用当前 RS-instructions 中的 RSICD 样本生成第一版训练集：

```bash
python scripts/prepare_data.py \
  --hf-dataset BigData-KSU/RS-instructions-dataset \
  --include-prefix RSICD \
  --path-map RSICD=$PWD/data/raw/RSICD \
  --out-dir data/rsllava
```

这里的 `--include-prefix RSICD` 很重要：如果只下载了 RSICD，却不加这个参数，其他来源的图片找不到，脚本会拒绝那些样本。

## RS-LLaVA 指令标注下载链接

官方 README 发布的原始 JSON 标注文件是：

<https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/resolve/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json?download=true>

注意官方文件名中的 `intrcutions` 拼写就是如此。下载后放到：

```text
data/annotations/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json
```

然后使用本地标注文件：

```bash
mkdir -p data/annotations
python scripts/prepare_data.py \
  --annotation data/annotations/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json \
  --include-prefix RSICD \
  --path-map RSICD=$PWD/data/raw/RSICD \
  --out-dir data/rsllava
```

## 直接从 Hugging Face 读取标注

在 `transformer` Conda 环境中执行：

```bash
conda activate transformer
cd /media/amers/SSD_part2/VLM/qwen3vl_rsllava_repro

python scripts/prepare_data.py \
  --hf-dataset BigData-KSU/RS-instructions-dataset \
  --hf-split train \
  --data-root /data/remote_sensing \
  --path-map RSICD=/data/remote_sensing/RSICD \
  --path-map UCM=/data/remote_sensing/UCM \
  --path-map UAV=/data/remote_sensing/UAV \
  --path-map RSVQA-LR=/data/remote_sensing/RSVQA-LR \
  --path-map RSIVQA-DOTA=/data/remote_sensing/RSIVQA-DOTA \
  --out-dir data/rsllava
```

这条命令会从 Hugging Face 读取标注，检查对应的本地图像，然后在本项目的 `data/rsllava/` 下生成可直接训练的 `train.json`、`validation.json` 和 `manifest.json`。如果希望同时生成 JSONL：

```bash
python scripts/prepare_data.py \
  --hf-dataset BigData-KSU/RS-instructions-dataset \
  --data-root /data/remote_sensing \
  --out-dir data/rsllava \
  --output-format both
```

也可以先把标注下载为本地 JSON，再使用 `--annotation` 和多个 `--path-map`。原图缺失时默认拒绝样本；`--allow-missing` 只适合检查标注格式，不应直接用于训练。

数据来源：

- [RS-instructions Dataset](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset)
- [RSICD images on Hugging Face](https://huggingface.co/datasets/arampacha/rsicd)
- [UCM-caption](https://pan.baidu.com/s/1mjPToHq#list/path=%2F)
- [RSVQA-LR](https://rsvqa.sylvainlobry.com/)
- [RSIVQA-DOTA](https://github.com/spectralpublic/RSIVQA)
- [RS-LLaVA repository](https://github.com/BigData-KSU/RS-LLaVA)
