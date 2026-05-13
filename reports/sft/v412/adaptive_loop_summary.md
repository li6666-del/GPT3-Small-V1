# V4.12 Adaptive 20-Round Summary

目标：按 harness v0.2 规则做 20 轮小步 SFT。每轮先写 strategy memo，再训练、评测、清理和决定 checkpoint 去留。

## Rounds

| round | focus | status | selected | init_after | summary |
| ---: | --- | --- | ---: | --- | --- |
| 0 | zh_factual_core | passed | 23 | `runs/sft-v412-00-zh_factual_core/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 1 | math_add | passed | 23 | `runs/sft-v412-01-math_add/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 2 | concept_ml | passed | 23 | `runs/sft-v412-02-concept_ml/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 3 | zh_factual_expand | failed | 16 | `runs/sft-v412-02-concept_ml/step_000023.pt` | Step 16 failed stage gates: simple_qa_boiling, simple_qa_week_days. |
| 4 | zh_factual_expand | failed | 16 | `runs/sft-v412-02-concept_ml/step_000023.pt` | Step 16 failed stage gates: simple_qa_boiling, simple_qa_week_days. |
| 5 | math_subtract | passed | 23 | `runs/sft-v412-05-math_subtract/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 6 | concept_science | passed | 23 | `runs/sft-v412-06-concept_science/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 7 | unknown_semantic | passed | 23 | `runs/sft-v412-07-unknown_semantic/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 8 | stop_short | passed | 23 | `runs/sft-v412-08-stop_short/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 9 | math_multiply | passed | 23 | `runs/sft-v412-09-math_multiply/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky. |
| 10 | practical_training | passed | 8 | `runs/sft-v412-10-practical_training/step_000008.pt` | Step 8 passed hard gates, with soft warnings: ability_simple_assistant. |
| 11 | short_explain | passed | 23 | `runs/sft-v412-11-short_explain/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |
| 12 | refusal_anchor | passed | 0 | `runs/sft-v412-12-refusal_anchor/step_000000.pt` | Step 0 passed hard gates, with soft warnings: ability_simple_assistant. |
| 13 | zh_core_consolidate | passed | 23 | `runs/sft-v412-13-zh_core_consolidate/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |
| 14 | zh_factual_core | passed | 23 | `runs/sft-v412-14-zh_factual_core/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |
| 15 | math_add | passed | 0 | `runs/sft-v412-15-math_add/step_000000.pt` | Step 0 passed hard gates, with soft warnings: ability_simple_assistant. |
| 16 | concept_ml | passed | 0 | `runs/sft-v412-16-concept_ml/step_000000.pt` | Step 0 passed hard gates, with soft warnings: ability_simple_assistant. |
| 17 | unknown_semantic | passed | 23 | `runs/sft-v412-17-unknown_semantic/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |
| 18 | zh_core_consolidate | passed | 23 | `runs/sft-v412-18-zh_core_consolidate/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |
| 19 | math_multiply | passed | 23 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 23 passed hard gates, with soft warnings: ability_simple_assistant. |

## Kept Checkpoints

- `runs/sft-v412-11-short_explain/step_000023.pt`
- `runs/sft-v412-17-unknown_semantic/step_000023.pt`
- `runs/sft-v412-18-zh_core_consolidate/step_000023.pt`
- `runs/sft-v412-19-math_multiply/step_000023.pt`

## Final Judgment

- 20 轮已完成：18 轮 passed，2 轮 failed。
- 失败集中在 `zh_factual_expand`：`一周有几天？` 和 `水在标准大气压下通常多少摄氏度沸腾？`。retry 后仍失败，因此后续不再硬推这个目标。
- `round12`、`round15`、`round16` 都选择了 step 0，说明这些目标在起点已经满足，没有新增训练收益；相关 checkpoint 已按冗余处理。
- 最终推荐继续点：`runs/sft-v412-19-math_multiply/step_000023.pt`。
- 仍未解决：`ability_simple_assistant`。最终输出能回答但不含“简单问题/简短解释”，因此继续作为 observe，不进入主线硬目标。

云端最终仅保留 V4.12 的 4 个权重：

```text
runs/sft-v412-11-short_explain/step_000023.pt
runs/sft-v412-17-unknown_semantic/step_000023.pt
runs/sft-v412-18-zh_core_consolidate/step_000023.pt
runs/sft-v412-19-math_multiply/step_000023.pt
```
