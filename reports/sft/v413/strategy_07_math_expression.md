# V4.13 Round 07 Strategy Memo

- focus: `math_expression`
- mode: `stage_retry`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `2.625e-07`
- reason: previous stage failed; retry same small target once: math_expression
- stage_rules: `math_add_1_4_exact, math_add_7_8`
- promoted_rules: `none`

## Previous Round

- status: `failed`
- accepted: `False`
- selected_step: `16`
- summary: Step 16 failed stage gates: math_add_1_4_exact.

## Decision

- Keep the step small and target one exposed weakness.
- Promote a repaired rule into future hard gates only after a non-zero selected checkpoint passes.
- Do not replace the V4.12 baseline unless the new checkpoint passes all active hard gates.
