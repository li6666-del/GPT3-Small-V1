# V4.14 Pre-Heldout Stabilization Summary

目标：在扩大中文 held-out 前做最后稳定化。每轮只处理一个暴露问题，跑满后由 best-step 决定是否保存。

## Rounds

| round | focus | status | accepted | selected | init_after | summary |
| ---: | --- | --- | --- | ---: | --- | --- |
| 0 | short_qa_corrections | passed | True | 36 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | Step 36 passed hard gates, with soft warnings: simple_qa_english_sky. |
| 1 | project_terms_short | failed | False | 0 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | Step 0 failed stage gates: practical_generation_eval, practical_valid_loss. |
| 2 | ability_plain | failed | False | 0 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | Step 0 failed stage gates: ability_simple_assistant. |
| 3 | preheldout_consolidate | failed | False | 0 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | Step 0 failed stage gates: ability_simple_assistant, practical_valid_loss. |

## Promoted Rules

- `math_add_1_4_exact`
- `simple_qa_week_days`

## Kept Checkpoints

- `runs/sft-v412-19-math_multiply/step_000023.pt`
- `runs/sft-v414-00-short_qa_corrections/step_000036.pt`

## Judgment

- V4.14 接受 `sft-v414-00-short_qa_corrections/step_000036.pt` 作为新的实验候选继续点。
- 本轮实际修复了两个强错误吸引子：`一周有几天？` 和 `1 加 4 等于多少？`。
- `project_terms_short`、`ability_plain` 和 `preheldout_consolidate` 都没有产生可保存 checkpoint；相关 `.pt` 已清理。
- 英文 sky 仍是 observe 失败，不进入中文主线硬门槛。
- V4.15 应该围绕 V4.14 step 36 做 pre-heldout regression，不应继续硬推项目术语和能力说明。
