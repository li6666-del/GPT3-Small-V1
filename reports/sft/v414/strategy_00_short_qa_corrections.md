# V4.14 Round 00 Strategy

- focus: `short_qa_corrections`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `1.2e-06`
- max_steps: `60`
- stage_rules: `simple_qa_week_days, math_add_1_4_exact`
- promoted_rules: `none`

决策：跑满本轮，不在 step 16 提前终止；只保存通过 main 和 stage 的非零 step。
