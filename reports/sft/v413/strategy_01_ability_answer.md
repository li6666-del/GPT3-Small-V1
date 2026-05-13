# V4.13 Round 01 Strategy Memo

- focus: `ability_answer`
- mode: `stage_retry`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `2.4000000000000003e-07`
- reason: previous stage failed; retry same small target once: ability_answer
- stage_rules: `ability_simple_assistant, ability_limit`
- promoted_rules: `none`

## Previous Round

- status: `failed`
- accepted: `False`
- selected_step: `16`
- summary: Step 16 failed stage gates: ability_simple_assistant.

## Decision

- Keep the step small and target one exposed weakness.
- Promote a repaired rule into future hard gates only after a non-zero selected checkpoint passes.
- Do not replace the V4.12 baseline unless the new checkpoint passes all active hard gates.
