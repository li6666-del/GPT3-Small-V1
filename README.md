# GPT3-Small-V1

一个从零训练的 125M decoder-only Transformer 项目，目标是做一个中英文小型 GPT 基座，并通过多轮 SFT 把它逐步调成能稳定按助手格式回答的模型。

截至 2026-05-13，项目已经完成：

- 125M GPT 基座预训练到 150k step。
- 选择 `runs/gpt3-small-125m/step_130000.pt` 作为 SFT 基座。
- 完成 V1 到 V4.7.1 多轮 SFT 实验。
- V4.7.1 在 V4.6.1 的短答、拒答、未知边界基础上，补入身份锚：`你是谁？` 会回答“驴肉火烧”的固定身份句。

完整过程、实验结论和后续计划见 [项目进程.md](项目进程.md)。

## 当前保留的关键模型

云端保留的关键 checkpoint：

| 用途 | 路径 | 说明 |
| --- | --- | --- |
| 基座 SFT 起点 | `runs/gpt3-small-125m/step_130000.pt` | 接近历史 best valid 的可用保存点 |
| V4.1 强格式锚点 | `runs/sft-v41-strong-format-from-130k/step_000150.pt` | 第一版明显学会短回答结束格式的 checkpoint |
| V4.3 最佳验证 | `runs/sft-v43-chinese-anchor-from-130k/step_000135.pt` | V4.3 best valid loss |
| V4.3 最终稳定版 | `runs/sft-v43-chinese-anchor-from-130k/step_000149.pt` | 早期 17 核心样本稳定锚点 |
| V4.5 近邻均衡版 | `runs/sft-v45-balanced-near-neighbor-from-v44/step_000125.pt` | V4.6 起点 |
| V4.6 主助手对齐 | `runs/sft-v46-assistant-alignment-from-v45/step_000200.pt` | best valid，作为 V4.6.1 回滚点保留 |
| V4.6.1 停止锚修复 | `runs/sft-v461-stop-anchor-repair-from-v46-step200/step_000020.pt` | V4.7 起点 |
| V4.7 身份预修复 | `runs/sft-v47-identity-boundary-from-v461-step20/step_000079.pt` | V4.7.1 起点，单独使用不推荐 |
| V4.7.1 身份强修复 | `runs/sft-v471-identity-force-from-v47-step79/step_000030.pt` | 当前推荐继续实验的 SFT checkpoint |

V1、V2、V3、V4、V4.2 的大权重文件已经不作为主线保留。V4.6 起每轮只保留必要 checkpoint。

## 项目结构

```text
configs/        训练配置，包括预训练和各轮 SFT 配置
data/           本地数据、SFT 样本和外部数据缓存
gpt_small/      模型、数据集、tokenizer、训练逻辑
logs/           训练日志
runs/           本地实验输出
scripts/        数据构建、下载、评测和生成脚本
README.md       快速说明
项目进程.md     项目主记录
```

## 模型配置

主配置文件：`configs/gpt3_small_125m.json`

- `vocab_size`: 50000
- `context_length`: 1024
- `n_layers`: 12
- `d_model`: 768
- `n_heads`: 12
- `d_ff`: 2048
- dropout: 0
- 参数量约 124M
- 训练精度：bf16

模型实现包括 decoder-only Transformer、SwiGLU、RoPE/causal attention、embedding/lm head 权重绑定等核心结构。

## 数据概览

预训练数据：

- English: `HuggingFaceFW/fineweb-edu`
- Chinese: `Morton-Li/ChineseWebText2.0-HighQuality`
- tokenizer: 50k BPE
- train tokens: 约 1.78B
- valid tokens: 约 22.3M

SFT 数据：

- V1/V2 主要是 synthetic bootstrap。
- V3 引入 Alpaca clean 和 Belle Chinese 数据。
- V4 系列改为小步课程式修正，重点修助手回答格式、中文锚定、拒绝、停止、短推理和翻译稳定性。

## 常用命令

预训练：

```powershell
python -m gpt_small.training.train --config configs/gpt3_small_125m.json
```

SFT：

```powershell
python -m gpt_small.training.sft --config configs/sft_125m_v471_identity_force.json
```

生成测试：

```powershell
python scripts/generate_text.py --config configs/gpt3_small_125m.json --checkpoint runs/gpt3-small-125m/step_130000.pt --prompt "你好，请介绍一下你自己。"
```

构建当前 SFT 数据：

```powershell
python scripts/build_sft_v46_dataset.py --out-dir data\sft\v46
python scripts/build_sft_v461_stop_repair_dataset.py --out-dir data\sft\v461_stop_repair
python scripts/build_sft_v47_identity_dataset.py --out-dir data\sft\v47_identity
python scripts/build_sft_v471_identity_force_dataset.py --out-dir data\sft\v471_identity_force
```

## 当前结论

V4.7.1 是当前主线结果。它不是通用助手模型，但已经能稳定回答身份类问题，同时保留短答、拒答、未知问题不编造、按要求停止等基础助手行为。

下一步建议是 V4.8：围绕身份锚做近邻泛化和短答边界修复，重点补 `你叫什么？只回答名字。`、`你能做什么？` 这类仍不稳的问法，不再扩大到大规模 SFT。
