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
| V4.7.1 身份强修复 | `runs/sft-v471-identity-force-from-v47-step79/step_000030.pt` | 保守主线基线 |
| V4.11-03 中文问答 micro | `runs/sft-v411-03-zh_qa-micro/step_000029.pt` | 通过主门槛，局部改善中文简单问答 |
| V4.11-04 数学 micro | `runs/sft-v411-04-math-micro/step_000029.pt` | 当前实验候选继续点，保留中文问答并补强 `2+3` |
| V4.12-17 unknown 语义 | `runs/sft-v412-17-unknown_semantic/step_000023.pt` | 窄中文路线中 unknown 语义边界通过点 |
| V4.12-18 中文核心巩固 | `runs/sft-v412-18-zh_core_consolidate/step_000023.pt` | 窄中文 QA / 数学巩固点 |
| V4.12-19 最终候选 | `runs/sft-v412-19-math_multiply/step_000023.pt` | 当前推荐继续点，20 轮自适应迭代最终通过点 |

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

V4.7.1 是保守主线基线。它不是通用助手模型，但已经能稳定回答身份类问题，同时保留短答、拒答、未知问题不编造、按要求停止等基础助手行为。

下一步建议是 V4.8：围绕身份锚做近邻泛化和短答边界修复，重点补 `你叫什么？只回答名字。`、`你能做什么？` 这类仍不稳的问法，不再扩大到大规模 SFT。

补充：V4.8/V4.8.1 已验证，普通身份格式修复和短答强修复都没有达到主线标准。当前推荐 checkpoint 仍是 V4.7.1 的 `runs/sft-v471-identity-force-from-v47-step79/step_000030.pt`。

补充：V4.10/V4.10.1/V4.11 已验证，broad assistant-core 和混合 core micro 都会相互污染；V4.11-04 是当前可继续实验的候选点，但还没有替代 V4.7.1 成为正式主线。

## SFT Harness

项目已加入最小可用 SFT Harness，用于把一轮 SFT 的固定流程自动化：

- 本地生成 SFT 数据。
- 上传数据、配置和脚本到云端。
- 启动云端训练。
- 定时拉取 `generation_eval.jsonl` 和日志。
- 按 YAML 中的 hard/soft gates 自动评测。
- 支持 `best_complete`，从所有完整 generation eval step 中选择最优 step，而不是只看最新 step。
- 支持 main / stage / observe 分层门槛，区分主线不可退化项、当前阶段目标和观察指标。
- 命中硬失败条件时自动 kill 训练。
- 生成 markdown 评测报告。
- 将失败原因写入 `reports/sft/failure_memory.jsonl`，供后续实验避坑。
- 通过 hard gates 后，可按 `iteration.continue_on_pass` 串联下一份实验 YAML。

示例：

```powershell
$env:AUTODL_PASSWORD="your_password"
python scripts/sft_harness.py --experiment experiments/sft_harness_boundary_example.yaml
```

注意：密码只走环境变量，不写入仓库。当前 harness 不会自己发明下一轮数据策略，只会执行 YAML 里明确指定的下一轮实验。

后续 harness v0.2 工作流约束：

- 每轮完成后，下一轮开始前，必须根据上一轮报告和 `reports/sft/failure_memory.jsonl` 调整策略。
- 每轮尽量只修一个主目标，最多一个辅助目标，保证步子小、成功率高、checkpoint 未来可用。
- checkpoint 是否保留由 best-step 评测和 Codex 判断共同决定：通过主门槛、阶段目标有收益、没有明显污染，才保留。

## 最新实验状态

2026-05-13 已用 SFT Harness 跑完 assistant-core、V4.11 micro-loop 和 V4.12 adaptive-loop 实践：

- V4.10 broad assistant-core：失败，已清理远程 `.pt`。
- V4.10.1 failure-memory repair：有明显局部改善，但仍失败，已清理远程 `.pt`。
- V4.11 十轮 micro-loop：完成并停止；2 轮通过主门槛，8 轮失败并清理 `.pt`。
- 云端当前仅保留 V4.11 的两个候选权重：
  - `runs/sft-v411-03-zh_qa-micro/step_000029.pt`
  - `runs/sft-v411-04-math-micro/step_000029.pt`
- V4.12 二十轮 adaptive-loop：完成并停止；18 轮 passed，2 轮 failed，失败集中在 `zh_factual_expand`。
- 云端当前保留 V4.12 的 4 个权重：
  - `runs/sft-v412-11-short_explain/step_000023.pt`
  - `runs/sft-v412-17-unknown_semantic/step_000023.pt`
  - `runs/sft-v412-18-zh_core_consolidate/step_000023.pt`
  - `runs/sft-v412-19-math_multiply/step_000023.pt`
- V4.13 中文修复轮：已跑 9 个小实验，全部未达到保存标准，云端 `.pt` 已清理。
  - `ability_answer` 会向拒答模板漂移。
  - `practical_terms` 对 `valid loss` 有局部改善，但仍半中半英，未过硬门槛。
  - `zh_week_days` 仍稳定错成“一周有 6 个月”。
  - `math_expression` 仍把 `1 加 4` 输出为 `4 + 4 = 5`。
- V4.14 pre-heldout stabilization：完成 4 个小实验，1 个 checkpoint 达到保存标准。
  - 保存：`runs/sft-v414-00-short_qa_corrections/step_000036.pt`
  - 已修复：`一周有几天？ -> 一周有 7 天。`
  - 已修复：`1 加 4 等于多少？ -> 1 + 4 = 5。`
  - 未修复：项目术语 `valid loss / generation_eval` 仍不干净。
  - 未修复：`你能做什么？` 仍不能稳定输出能力说明。
- V4.15 ability must-fix：完成 5 个小实验，最终 checkpoint 达到保存标准。
  - 保存：`runs/sft-v415-04-core_regression_repair/step_000043.pt`
  - 已修复：`你能做什么？`、`你的能力是什么？`、`你可以帮我做什么？`
  - 已守住：身份、stop、拒答、unknown、`1+4`、一周七天、`9-4`、沸腾点。
  - 未修复：英文 sky，继续作为 observe。
- V4.16 real Chinese probe：完成 1 轮 eval-only 和 2 轮小步 SFT，未产生可保存 checkpoint。
  - 新增 harness eval-only：可直接评测 checkpoint，不训练。
  - 真实中文 probe 显示：身份、拒答、stop、核心中文常识、基础算术稳定。
  - 暴露短板：ability 近邻、unknown 近邻、project terms。
  - V4.16 两轮修复都未达到保存标准，新 `.pt` 已清理。
- V4.17 中文 held-out v1：完成 242 条正式 held-out 基线评测，不训练、不产生 checkpoint。
  - 类别通过率：refusal 0.88、stop 0.88、identity 0.70、ability 0.69、unknown 0.64、qa 0.40、math 0.33、project_terms 0.31。
  - 结论：V4.15 已具备部分助手外壳，但还不是可靠简单问答模型。
  - 下一步不建议直接扩 broad QA / math；优先小步修 identity short-name 和 stop exact，同时守住 refusal / unknown。
- V4.17.1-V4.17.8 小步迭代：完成 8 轮，最终保存 1 个当前候选 checkpoint。
  - 保存：`runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt`
  - 已验证：identity fresh、unknown fresh、refusal fresh、stop semantic 和核心 QA/math anchors 过 main/stage gate。
  - 已降级：ability fresh、project terms、broad QA 变体、strict stop exact 作为 observe，不阻塞 checkpoint。
  - harness 改进：修复 `equals_expected` expected 字段合并；改进 `best_complete`，优先选择 gate 通过率更好的 step。

结论：当前保守基线仍是 `runs/sft-v471-identity-force-from-v47-step79/step_000030.pt`；当前实验候选继续点更新为 `runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt`。下一步可以进入正式 held-out，但必须分 main / stage / observe，避免 broad QA、泛化算术和 project_terms 阻塞主线 checkpoint。
