# GPT3-small-V1 Project Memory

更新时间: 2026-05-11

## 项目位置

本地:

```text
D:\GPT3 small V1
```

GitHub:

```text
https://github.com/li6666-del/GPT3-Small-V1.git
```

AutoDL:

```text
SSH: ssh -p 29372 root@connect.bjb2.seetacloud.com
工作目录: /root/autodl-tmp/GPT3-small-V1
持久目录: /root/autodl-fs/GPT3-small-V1
```

AutoDL 项目目录里有软链接:

```text
data        -> /root/autodl-fs/GPT3-small-V1/data
artifacts   -> /root/autodl-fs/GPT3-small-V1/artifacts
checkpoints -> /root/autodl-fs/GPT3-small-V1/checkpoints
logs        -> /root/autodl-fs/GPT3-small-V1/logs
```

## 当前代码能力

已实现:

```text
decoder-only TransformerLM
SwiGLU MLP
memmap token dataset
AMP / gradient accumulation / checkpoint resume
JSONL training log
token-level generation
byte-level BPE tokenizer trainer/tokenizer
txt -> token bin
Hugging Face parquet cache downloader
Windows local dataset downloader
SFTP upload helper
pretraining data preview extractor
```

主模型配置:

```text
config: configs/gpt3_small_125m.json
vocab_size: 50000
context_length: 1024
num_layers: 12
d_model: 768
num_heads: 12
d_ff: 2048
activation: SwiGLU
parameters: 124,159,488
```

## 数据方案

坚持使用最初决定的数据:

```text
英文: HuggingFaceFW/fineweb-edu, sample/10BT
中文: Morton-Li/ChineseWebText2.0-HighQuality
比例: 英文 50% / 中文 50%
阶段 1 目标: train 约 1B tokens, valid 约 10M tokens
```

本地已下载 parquet:

```text
总 parquet: 94 个
总大小: 36.02 GiB
英文: 14 个, 26.56 GiB
中文: 80 个, 9.46 GiB
```

本地路径:

```text
data/cache/fineweb_edu_10bt/sample/10BT/*.parquet
data/cache/chinesewebtext2_hq/data/*.parquet
```

预览文件:

```text
data/samples/pretrain_preview.txt
```

观察:

```text
英文 FineWeb-Edu 有少量 mojibake, 例如 Jane Austen鈥檚
中文 ChineseWebText2 可读, 但有网页公告、财经、法律文本、广告句等常见网页语料噪声
后续正式生成 raw text 前建议加轻量清洗/过滤
```

## 上传状态

本地到 AutoDL 的 SFTP 上传脚本:

```text
scripts/upload_cache_to_autodl.py
```

上传日志:

```text
logs/cache_upload_to_autodl_resume.log
```

截至本文件更新时间:

```text
远端 parquet: 92 / 94
远端大小: 约 33G
中文: 80 / 80 完成
英文: 12 / 14 在远端
正在补传英文 fineweb_edu_10bt/sample/10BT/011_00000.parquet 附近
```

如果上传中断, 重新运行上传脚本即可:

```powershell
$env:AUTODL_PASSWORD="..."
python scripts/upload_cache_to_autodl.py --local-root data/cache --remote-root /root/autodl-fs/GPT3-small-V1/data/cache
```

脚本会跳过远端已完整文件, 对半截文件会覆盖重传该文件。

## 下一步

1. 等上传完成后, 验证远端:

```text
parquet 总数应为 94
英文应为 14
中文应为 80
```

2. 在 AutoDL 上从本地 parquet 生成 raw text:

```bash
cd /root/autodl-tmp/GPT3-small-V1
python scripts/prepare_pretrain_text.py \
  --en-local-glob "data/cache/fineweb_edu_10bt/**/*.parquet" \
  --zh-local-glob "data/cache/chinesewebtext2_hq/**/*.parquet" \
  --train-bytes 8000000000 \
  --valid-bytes 100000000
```

3. 训练 50k byte-level BPE tokenizer:

```bash
python -m gpt_small.tokenizer.bpe_trainer \
  --input-path data/raw/train.txt \
  --vocab-size 50000
```

4. tokenize:

```bash
python scripts/tokenize_corpus.py --input-path data/raw/train.txt --output-path data/tokens/train.bin
python scripts/tokenize_corpus.py --input-path data/raw/valid.txt --output-path data/tokens/valid.bin
```

5. 正式训练前检查:

```text
vocab size 是否为 50000
config 中 data dtype 是否为 uint16
GPU / torch CUDA 是否可用
checkpoint/log 输出是否落到持久盘
```

## 协作约定

opencode 只在用户明确要求时调用。

默认由 Codex 直接执行任务, 不主动咨询本地或远端 opencode, 避免超时浪费时间。
