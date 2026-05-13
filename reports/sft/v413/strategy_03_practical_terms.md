# V4.13 Round 03 Strategy Memo

- focus: `practical_terms`
- mode: `stage_retry`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `2.4000000000000003e-07`
- reason: previous stage failed; retry same small target once: practical_terms
- stage_rules: `practical_valid_loss, practical_heldout`
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
