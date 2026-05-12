# GPT3-Small-V1

一个从零训练的 125M decoder-only Transformer 项目，目标是做一个中英文小型 GPT 基座，并通过多轮 SFT 把它逐步调成能稳定按助手格式回答的模型。

截至 2026-05-12，项目已经完成：

- 125M GPT 基座预训练到 150k step。
- 选择 `runs/gpt3-small-125m/step_130000.pt` 作为 SFT 基座。
- 完成 V1 到 V4.3 多轮 SFT 实验。
- V4.3 在短回答、拒绝、数学、翻译、停止输出等核心固定样本上，已经能通过 greedy 和多 seed 抽样检查。

完整过程、实验结论和后续计划见 [项目进程.md](项目进程.md)。

## 当前保留的关键模型

云端保留的关键 checkpoint：

| 用途 | 路径 | 说明 |
| --- | --- | --- |
| 基座 SFT 起点 | `runs/gpt3-small-125m/step_130000.pt` | 接近历史 best valid 的可用保存点 |
| V4.1 强格式锚点 | `runs/sft-v41-strong-format-from-130k/step_000150.pt` | 第一版明显学会短回答结束格式的 checkpoint |
| V4.3 最佳验证 | `runs/sft-v43-chinese-anchor-from-130k/step_000135.pt` | V4.3 best valid loss |
| V4.3 最终稳定版 | `runs/sft-v43-chinese-anchor-from-130k/step_000149.pt` | 当前推荐继续实验的 SFT checkpoint |

V1、V2、V3、V4、V4.2 的大权重文件已经不作为主线保留，相关日志和评测结论合并进项目进程文档。

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
python -m gpt_small.training.sft --config configs/sft_125m_v43.json
```

生成测试：

```powershell
python scripts/generate_text.py --config configs/gpt3_small_125m.json --checkpoint runs/gpt3-small-125m/step_130000.pt --prompt "你好，请介绍一下你自己。"
```

构建 V4.3 数据：

```powershell
python scripts/build_sft_v43_dataset.py
```

## 当前结论

V4.3 是目前第一版真正通过“固定核心样本 + greedy + 多 seed”检查的 SFT 结果。它还不是通用助手模型，但已经证明：在 130k 基座上，小步、强格式、中文锚点、短输出约束是有效路线。

下一步建议是 V4.4：围绕 V4.3 已经稳定的能力做近邻扩展，不急着扩大到泛化型大 SFT。
