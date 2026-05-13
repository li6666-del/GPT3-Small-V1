# V4.13 Round 00 Strategy Memo

- focus: `ability_answer`
- mode: `normal`
- init_checkpoint: `runs/sft-v412-19-math_multiply/step_000023.pt`
- learning_rate: `3.2e-07`
- reason: start V4.13 Chinese repair loop
- stage_rules: `ability_simple_assistant, ability_limit`
- promoted_rules: `none`

## Previous Round

- none

## Decision

- Keep the step small and target one exposed weakness.
- Promote a repaired rule into future hard gates only after a non-zero selected checkpoint passes.
- Do not replace the V4.12 baseline unless the new checkpoint passes all active hard gates.
