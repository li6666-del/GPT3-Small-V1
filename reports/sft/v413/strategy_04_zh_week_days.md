# V4.13 Round 04 Strategy Memo

- focus: `zh_week_days`
- mode: `skip_failed_target`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `3.5e-07`
- reason: previous target failed after retry budget; move to next small target
- stage_rules: `simple_qa_week_days`
- promoted_rules: `none`

## Previous Round

- status: `failed`
- accepted: `False`
- selected_step: `16`
- summary: Step 16 failed stage gates: practical_valid_loss.

## Decision

- Keep the step small and target one exposed weakness.
- Promote a repaired rule into future hard gates only after a non-zero selected checkpoint passes.
- Do not replace the V4.12 baseline unless the new checkpoint passes all active hard gates.
