# V4.14 Round 03 Strategy

- focus: `preheldout_consolidate`
- init_checkpoint: `runs/sft-v414-00-short_qa_corrections/step_000036.pt`
- learning_rate: `3.2e-07`
- max_steps: `36`
- stage_rules: `simple_qa_week_days, math_add_1_4_exact, practical_valid_loss, ability_simple_assistant`
- promoted_rules: `math_add_1_4_exact, simple_qa_week_days`

决策：跑满本轮，不在 step 16 提前终止；只保存通过 main 和 stage 的非零 step。
