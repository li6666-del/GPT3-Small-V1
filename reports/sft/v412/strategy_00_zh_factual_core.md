# V4.12 Round 00 Strategy Memo

- focus: `zh_factual_core`
- mode: `normal`
- init_checkpoint: `runs/sft-v411-04-math-micro/step_000029.pt`
- learning_rate: `4.5e-07`
- reason: start narrow Chinese assistant loop
- stage_rules: `simple_qa_china, simple_qa_months`

## Previous Round

- none

## Failure Memory Top Rules

- `unknown_no_certain_checkpoint`: 7
- `ability_simple_assistant`: 7
- `simple_qa_english_sky`: 7
- `unknown_no_fabrication_explicit`: 5
- `identity_not_chatgpt`: 1
- `simple_math_add`: 1
- `unknown_no_fabrication`: 1

## Decision

- Keep the round small: one focus, no English hard target, ability stays observe unless explicitly selected later.
- Preserve identity, stop, refusal, unknown safe, H2O, France, and 2+3 as main regression.
- Save checkpoint only if main gates and stage gates pass; otherwise delete `.pt` and keep report only.
