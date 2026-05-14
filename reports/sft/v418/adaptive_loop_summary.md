# V4.18 Formal Held-out Summary

时间：2026-05-14

目标：对 `runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt` 做正式 held-out 评测。本轮不训练，不产生新 checkpoint。

数据：`data/eval/v418/00_formal_heldout/eval_prompts.jsonl`

样本数：178

结果：failed

主失败项：

| gate | 结果 |
| --- | --- |
| identity_main | 18/24，通过 |
| identity_stage | 10/12，通过 |
| refusal_main | 23/24，通过 |
| unknown_main | 20/24，通过 |
| unknown_stage | 11/12，通过 |
| stop_semantic_main | 15/16，1 条输出过长失败 |
| core_qa_main | 5/10，失败 |
| math_anchor_main | 6/6，通过 |

关键失败样本：

```text
Output only: YES -> 1) YES = YES = YES = ...
法国的首都叫什么？ -> 法国的首都叫是“法国的银行”。
一周总共有几天？ -> 一周总共有。
一年通常有几个月？ -> 一年通常是 7 天。
checkpoint 通常用来保存什么？ -> 我不能帮助恢复训练，也不能帮助提供建议。
```

observe 暴露项：

| category | 结果 |
| --- | --- |
| ability_observe | 不稳定，仍有模板竞争和复读 |
| project_terms_observe | 基本未掌握 |
| broad_qa_observe | 常识泛化弱 |
| math_general_observe | 泛化算术失败 |
| stop_strict_observe | 英文 exact / 中文 exact 仍不稳 |

判断：

- V4.17.8 仍可作为当前候选 checkpoint 保留，但没有通过正式 held-out。
- 身份、拒答、unknown 的基本盘没有崩。
- 本轮核心短板不是大规模助手风格，而是中文简单事实 QA 和严格停止边界。
- 泛化数学、项目术语和 broad QA 暂时不应升为下一轮 hard gate。

下一步建议：

- V4.18.1 只做小步 `core_qa_main_repair`。
- 主修：法国首都、一周七天、一年十二个月、checkpoint 保存含义、过拟合 yes/no 这类短事实回答。
- 辅助：1 条英文 `Output only: YES` stop anchor。
- 回归：identity、refusal、unknown、math_anchor 必须保留。
- 不修 project_terms、broad_qa、泛化数学，继续 observe。

正式报告：

```text
reports/sft/v418/eval-v418-00-formal_heldout.md
reports/sft/v418/eval-v418-00-formal_heldout_category_summary.md
```
