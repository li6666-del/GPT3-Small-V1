# GPT-3 Small V1

这是一个用于研究 GPT-3 Small 规模中英文 base model 的工程实验项目。

第一阶段目标是先搭好训练基础设施：

- decoder-only TransformerLM
- memmap token 数据集
- AMP 混合精度
- 梯度累积
- checkpoint 保存和恢复训练
- JSONL 训练日志
- 简单的自回归生成
- byte-level BPE tokenizer 训练和语料 tokenization

## 模型配置

当前主配置文件是 `configs/gpt3_small_125m.json`，目标是 GPT-3 Small 级别的
decoder-only language model：

```text
vocab_size: 50000
context_length: 1024
num_layers: 12
d_model: 768
num_heads: 12
d_ff: 2048
activation: SwiGLU
dropout: 0.0
parameters: 约 124M
```

这里的前馈网络使用 SwiGLU。为了让参数量仍然接近 GPT-3 Small 的 125M 级别，
`d_ff` 使用 2048，而不是传统 GELU MLP 常见的 3072。

## 快速 Smoke Test

```powershell
python scripts/make_toy_memmap.py --out-dir data/toy --vocab-size 256 --tokens 20000
python -m gpt_small.training.train --config configs/smoke.json
python -m gpt_small.generate --checkpoint runs/smoke/latest.pt --prompt "1 2 3"
```

`smoke` 配置刻意做得很小，可以在 CPU 上运行。真实训练可以从
`configs/gpt3_small_125m.json` 开始调整。

## Tokenizer

CS336 阶段的 byte-level BPE tokenizer 已迁移到：

```text
gpt_small/tokenizer/bpe_trainer.py
gpt_small/tokenizer/bpe_tokenizer.py
scripts/tokenize_corpus.py
```

训练 tokenizer：

```powershell
python -m gpt_small.tokenizer.bpe_trainer --input-path data/raw/corpus.txt --vocab-size 50000
```

默认产物位置：

```text
artifacts/tokenizer/vocab.bin
artifacts/tokenizer/merges.bin
```

把文本语料编码成训练用 memmap token 文件：

```powershell
python scripts/tokenize_corpus.py --input-path data/raw/train.txt --output-path data/tokens/train.bin
python scripts/tokenize_corpus.py --input-path data/raw/valid.txt --output-path data/tokens/valid.bin
```

## 数据准备

第一版预训练数据方案：

```text
英文: HuggingFaceFW/fineweb-edu, sample-10BT
中文: Morton-Li/ChineseWebText2.0-HighQuality
比例: 英文 50% / 中文 50%
阶段 1 目标: train 约 1B tokens, valid 约 10M tokens
```

先用 streaming 准备原始文本：

```powershell
python scripts/prepare_pretrain_text.py --train-bytes 8000000000 --valid-bytes 100000000
```

这会生成：

```text
data/raw/train.txt
data/raw/valid.txt
```

在 AutoDL 上建议先把 parquet 分片下载到持久盘缓存，再从本地 parquet 生成文本：

```bash
python scripts/download_hf_parquets.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --prefix sample/10BT \
  --output-dir data/cache/fineweb_edu_10bt \
  --max-files 14

python scripts/download_hf_parquets.py \
  --dataset Morton-Li/ChineseWebText2.0-HighQuality \
  --prefix data \
  --output-dir data/cache/chinesewebtext2_hq \
  --max-files 80

python scripts/prepare_pretrain_text.py \
  --en-local-glob "data/cache/fineweb_edu_10bt/**/*.parquet" \
  --zh-local-glob "data/cache/chinesewebtext2_hq/**/*.parquet" \
  --train-bytes 8000000000 \
  --valid-bytes 100000000
```
