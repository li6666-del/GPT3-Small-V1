# V4.12 Round 01 Strategy Memo

- focus: `math_add`
- mode: `normal`
- init_checkpoint: `runs/sft-v412-00-zh_factual_core/step_000023.pt`
- learning_rate: `5e-07`
- reason: previous checkpoint accepted; advance to next small target
- stage_rules: `math_add_1_4, math_add_7_8`

## Previous Round

- status: `passed`
- selected_step: `23`
- summary: Step 23 passed hard gates, with soft warnings: ability_simple_assistant, simple_qa_english_sky.

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
