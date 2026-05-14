# GPT3-Small-V1

从零训练的 124M decoder-only Transformer 小模型项目。项目目标不是追求大模型能力，而是完整走通 LLM 工程流程：预训练、SFT、评测、checkpoint 选择、失败复盘和迭代管理。

当前阶段：SFT。已停止继续预训练或二次预训练，后续只围绕“可用小助手”能力做小步 SFT。

## 当前状态

- 基座模型：125M 级 GPT，小模型实际参数约 124M。
- 预训练：已完成到 150k step，历史 SFT 起点选择 `runs/gpt3-small-125m/step_130000.pt`。
- 当前最佳 SFT checkpoint：
  - 云端：`runs/sft-v4189-00-usable_core_checkpoint/step_000055.pt`
  - 本地：`checkpoints/sft-v4189-00-usable_core_checkpoint/step_000055.pt`
- 当前权重保留策略：只保留 V4.18.9 最佳权重；旧 SFT 权重不再保留。
- 当前结论：V4.18.9 是目前最稳的可用助手基线。V4.18.10-V4.18.12 尝试修 unknown 表达，但会引入算术或表达退化，未保存权重。

## 模型能力

V4.18.9 已通过的核心门槛：

- 身份回答稳定：能说明自己是 124M 小语言模型，名字是“驴肉火烧”。
- 能否认自己是 OpenAI 官方模型或 ChatGPT。
- 能做基础能力说明。
- 能拒绝盗号、偷账号等危险请求。
- 能按要求输出短词，例如 `完成`、`结束`、`YES`。
- 能回答少量固定基础 QA：法国首都、北京、H2O。
- 能回答少量固定算术：`1 + 4 = 5`、`5 × 8 = 40`。

仍然不可靠的方向：

- 泛化算术，尤其是换问法后的加减乘。
- 时间单位问题，例如“一周几天”“一年几个月”。
- project terms，例如 checkpoint / generation_eval / valid loss 的解释。
- unknown 边界的自然表达，容易混入项目内部话术。
- 更宽的中文 held-out 问答。

## 项目结构

```text
configs/        训练和评测配置
experiments/    SFT harness 实验 YAML
gpt_small/      模型、数据、tokenizer、训练代码
scripts/        数据处理、训练辅助、harness、评测脚本
reports/sft/    failure_memory.jsonl，保留失败经验供 harness 复盘
README.md       项目快速说明
项目进程.md     当前进展和决策记录
```

本地大文件目录：

```text
data/           本地数据缓存，Git 忽略
runs/           本地训练输出，Git 忽略
checkpoints/    本地保留 checkpoint，Git 忽略
logs/           本地日志，Git 忽略
```

## 常用命令

预训练入口：

```powershell
python -m gpt_small.training.train --config configs/gpt3_small_125m.json
```

SFT 训练入口：

```powershell
python -m gpt_small.training.sft --config configs/sft_125m_v4178_00_preheldout_mainline_gate.json
```

Harness 入口：

```powershell
$env:AUTODL_PASSWORD="your_password"
python scripts/sft_harness.py --experiment experiments/v4178/sft_v4178_00_preheldout_mainline_gate.yaml
```

固定 checkpoint 生成评测：

```powershell
python scripts/checkpoint_generation_eval.py --config <eval_config.json>
```

## Harness 原则

- 每轮 SFT 前读取上一轮报告和 `reports/sft/failure_memory.jsonl`。
- 每轮只主修一个能力，最多带一个辅助能力。
- 身份、stop、拒答、unknown 是长期 regression。
- 不按最新 step 或最低 loss 自动保存。
- 只有 main gates 不退化、目标有明确收益、无明显污染时才保留 checkpoint。
- 失败轮次删除 `.pt`，只保留结论进入文档和 failure memory。

## 当前建议

以 V4.18.9 作为当前可用基线。下一阶段不要继续强修时间单位或 project terms；如果继续 SFT，应先保护 `1+4`、身份、stop、拒答，再小步修 unknown 的自然表达。
