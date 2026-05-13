# V4.14 Round 01 Strategy

- focus: `project_terms_short`
- init_checkpoint: `runs/sft-v414-00-short_qa_corrections/step_000036.pt`
- learning_rate: `7e-07`
- max_steps: `60`
- stage_rules: `practical_valid_loss, practical_generation_eval, practical_heldout_worse`
- promoted_rules: `math_add_1_4_exact, simple_qa_week_days`

决策：跑满本轮，不在 step 16 提前终止；只保存通过 main 和 stage 的非零 step。
