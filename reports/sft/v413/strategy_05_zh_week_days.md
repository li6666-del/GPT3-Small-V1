# V4.13 Round 05 Strategy Memo

- focus: `zh_week_days`
- mode: `stage_retry`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `2.625e-07`
- reason: previous stage failed; retry same small target once: zh_week_days
- stage_rules: `simple_qa_week_days`
- promoted_rules: `none`

## Previous Round

- status: `failed`
- accepted: `False`
- selected_step: `8`
- summary: Step 8 failed stage gates: simple_qa_week_days.

## Decision

- Keep the step small and target one exposed weakness.
- Promote a repaired rule into future hard gates only after a non-zero selected checkpoint passes.
- Do not replace the V4.12 baseline unless the new checkpoint passes all active hard gates.
