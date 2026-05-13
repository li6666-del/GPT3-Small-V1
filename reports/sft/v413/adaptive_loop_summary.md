# V4.13 Chinese Repair Adaptive Summary

目标：在 V4.12-19 基本盘上修复中文能力说明、常识短答、算术表达和项目术语乱码。每轮只推进一个小目标；通过后把该目标提升为后续硬门槛。

## Rounds

| round | focus | status | accepted | selected | init_after | summary |
| ---: | --- | --- | --- | ---: | --- | --- |
| 0 | ability_answer | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: ability_simple_assistant. |
| 1 | ability_answer | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: ability_limit, ability_simple_assistant. |
| 2 | practical_terms | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: practical_valid_loss. |
| 3 | practical_terms | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: practical_valid_loss. |
| 4 | zh_week_days | failed | False | 8 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 8 failed stage gates: simple_qa_week_days. |
| 5 | zh_week_days | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: simple_qa_week_days. |
| 6 | math_expression | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: math_add_1_4_exact. |
| 7 | math_expression | failed | False | 16 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 16 failed stage gates: math_add_1_4_exact. |
| 8 | practical_terms_full | failed | False | 43 | `runs/sft-v412-19-math_multiply/step_000023.pt` | Step 43 failed stage gates: practical_valid_loss. |

## Promoted Rules

- none

## Kept Checkpoints

- `runs/sft-v412-19-math_multiply/step_000023.pt`

## Judgment

- 本轮没有产生可保存的新 checkpoint，继续保留 V4.12-19 作为主线。
- 失败不是主线硬门槛退化：身份、stop、拒答、unknown、H2O、法国首都、2+3 在各轮仍通过。
- 失败集中在新目标：能力说明、`valid loss` 项目术语、一周七天、`1 + 4` 表达一致。
- `practical_terms_full` 跑满 44 step 后仍未过关，说明仅靠重复 SFT 样本不能可靠修复该点。
- 下一阶段应降低单点硬推强度，改为“中文短答模板 + 更短输出 + 更小学习率 + 不把能力说明和拒答样本混在同一小轮”。
