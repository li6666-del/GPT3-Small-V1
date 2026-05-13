# V4.15 Round 00 Strategy

- focus: `ability_acquire_balanced`
- init_checkpoint: `runs/sft-v414-00-short_qa_corrections/step_000036.pt`
- learning_rate: `7e-07`
- max_steps: `64`
- cleanup_enabled: `False`
- stage_rules: `ability_simple_assistant, ability_capability, ability_can_help, ability_brief, ability_limit`

目标：能力说明必须通过，同时守住 V4.14 已修复的 `1+4` 和一周七天。
