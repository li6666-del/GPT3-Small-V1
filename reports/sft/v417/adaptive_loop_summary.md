# V4.17 中文 held-out v1 基线评测总结

时间：2026-05-14

## 本轮性质

V4.17 不是训练轮，而是正式中文 held-out v1 的基线评测轮。

目的：

- 用 V4.15 当前候选 checkpoint 建立更大的中文 held-out 基线。
- 验证 V4.15 到底能泛化到哪些中文助手行为。
- 为后续小步 SFT 选择主修方向，避免继续凭少量固定样本判断模型能力。

评测起点：

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
```

## 数据与流程

新增 held-out v1：

```text
data/eval/v417/00_zh_heldout_v1/eval_prompts.jsonl
```

样本数：242 条。

类别：

- identity：30
- ability：36
- unknown：36
- refusal：24
- stop：24
- qa：40
- math：36
- project_terms：16

执行方式：

- 本地生成 held-out eval 数据。
- 本地执行 JSONL 文本审计，确认无 mojibake 乱码。
- 上传云端。
- 通过 harness eval-only 模式直接评测 checkpoint。
- 生成 category summary 和正式 harness report。

## 结果

| category | pass | total | rate |
| --- | ---: | ---: | ---: |
| ability | 25 | 36 | 0.69 |
| identity | 21 | 30 | 0.70 |
| math | 12 | 36 | 0.33 |
| project_terms | 5 | 16 | 0.31 |
| qa | 16 | 40 | 0.40 |
| refusal | 21 | 24 | 0.88 |
| stop | 21 | 24 | 0.88 |
| unknown | 23 | 36 | 0.64 |

harness 主报告状态为 `failed`，主要原因是 stop exact gate 过严且确实存在若干真正停止失败。

需要区分两类 stop 失败：

- 语义可接受但 exact 不通过：例如 `结束` 输出成 `结束。`。
- 真正失败：例如要求只输出 `YES`，却输出无关内容；要求只回答 `知道了`，却回答 `明白了。`

后续 gate 应拆成：

- `stop_semantic`：主线门槛。
- `stop_exact_strict`：观察指标或阶段门槛。

## 主要发现

V4.15 的强项：

- 拒答基本稳定，held-out 通过率 0.88。
- stop 的宽松语义基本稳定，held-out 通过率 0.88。
- 身份、能力、unknown 已有一定近邻能力，但距离可靠助手仍不足。
- 固定训练锚样本仍稳定。

V4.15 的短板：

- 中文常识 QA 泛化弱，held-out 通过率 0.40。
- 算术泛化弱，held-out 通过率 0.33。
- 项目术语泛化弱，held-out 通过率 0.31。
- 身份短问法仍不稳，例如 `你叫什么名字？`、`你的昵称是什么？`。
- ability 和 unknown 之间仍存在模板竞争。

## checkpoint 决策

本轮不训练，不产生新 checkpoint。

继续保留 V4.15：

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
```

不恢复 V4.16 的任何 checkpoint，因为 V4.16 小修未达到保存标准。

## 下一步策略

不建议下一轮直接扩大中文 QA 或算术。

原因：

- held-out 已证明 broad QA / math 是当前 124M 模型低收益、高污染风险方向。
- 算术问题更接近预训练缺口，不适合用主线 SFT 硬补。
- 当前更有收益的是修助手外壳：身份短问法、exact stop、unknown 不编造。

下一轮建议：

- 主修：identity short-name + stop exact repair。
- 辅助：保持 refusal / unknown regression。
- 不训练 held-out 原题，只生成近邻开发样本。
- 继续从 V4.15 checkpoint 起步。

保存标准：

- 身份短问法明显改善。
- 真正 stop 失败减少。
- refusal / unknown / 基础 QA / 已有算术锚不退化。
- 不出现身份模板污染 ability 或 unknown。
