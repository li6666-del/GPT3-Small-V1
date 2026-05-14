# V4.17.1-V4.17.8 小步迭代总结

时间：2026-05-14

## 目标

按 harness 流程继续小步迭代，原则是稳、扬长避短，为下一阶段正式 held-out 做准备。

起点：

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
```

最终当前候选：

```text
runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt
```

## harness 改进

本轮修了两个 workflow 问题：

- `equals_expected` 评测现在会把 eval prompt 的 `expected` 字段合并进 generation rows，避免 stop exact 被误判。
- `best_complete` 不再只按 hard gate 数量和 latest step 选点；当失败规则数量相同，会优先选择通过率更高的 step。

## 迭代过程

| round | 目标 | 结果 | 决策 |
| --- | --- | --- | --- |
| V4.17.1 | identity short + stop exact | 失败；identity 3/10，stop exact 修正后 5/12 | 不保存 |
| V4.17.2 | identity short only | 失败；identity 提升到 5/10，但不稳定 | 不保存 |
| V4.17.3 | identity dev-hard | 失败；强压短答反而退化 | 不保存 |
| V4.17.4 | identity template redirect | 通过；step 8 保存 | 保存后作为中间起点 |
| V4.17.5 | fresh pre-heldout eval | 失败；identity/stop 过，refusal 被身份模板污染 | 不训练 |
| V4.17.6 | refusal repair | 拒答修好，但 ability/unknown 被挤压，QA 变体阻塞 | 不保存 |
| V4.17.7 | balanced consolidate | main 过，ability fresh 卡住 | 不保存 |
| V4.17.8 | mainline gate | hard gates 全过，ability/project terms/broad QA 作为 observe | 保存 |

## 当前能力判断

V4.17.8 已经守住：

- identity fresh：9/12
- unknown fresh：5/6
- refusal fresh：5/5
- stop semantic：6/6
- 核心 QA/math anchors：主线样本通过

仍作为 observe 的短板：

- ability fresh：3/6
- project_terms：0/3
- broad QA 变体：`标准大气压下水多少度沸腾？` 仍失败

## checkpoint 决策

保留：

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt
```

删除：

```text
runs/sft-v4174-00-identity_template_redirect/step_000008.pt
```

理由：V4.17.8 已吸收 V4.17.4 的身份收益，并修回 refusal；V4.17.4 不再是必要分支。

## 下一步

我判断现在可以进入正式 held-out。

下一阶段 held-out 应该明确分层：

- main：identity、refusal、unknown、stop semantic、核心 QA/math anchors。
- stage：少量 near-neighbor identity/unknown。
- observe：ability fresh、project terms、broad QA、strict stop exact、泛化算术。

不要把 broad QA / 泛化算术 / project_terms 作为下一轮保存阻塞项，否则会重新把训练带回低收益方向。

