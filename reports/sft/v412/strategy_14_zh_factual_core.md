# V4.12 Round 14 Strategy Memo

- focus: `zh_factual_core`
- mode: `normal`
- init_checkpoint: `runs/sft-v412-13-zh_core_consolidate/step_000023.pt`
- learning_rate: `4.5e-07`
- reason: previous checkpoint accepted; advance to next small target
- stage_rules: `simple_qa_china, simple_qa_months`

## Previous Round

- status: `passed`
- selected_step: `23`
- summary: Step 23 passed hard gates, with soft warnings: ability_simple_assistant.

## Failure Memory Top Rules

- `unknown_no_certain_checkpoint`: 7
- `ability_simple_assistant`: 7
- `simple_qa_english_sky`: 7
- `unknown_no_fabrication_explicit`: 5
- `simple_qa_boiling`: 2
- `simple_qa_week_days`: 2
- `identity_not_chatgpt`: 1
- `simple_math_add`: 1

## Decision

- Keep the round small: one focus, no English hard target, ability stays observe unless explicitly selected later.
- Preserve identity, stop, refusal, unknown safe, H2O, France, and 2+3 as main regression.
- Save checkpoint only if main gates and stage gates pass; otherwise delete `.pt` and keep report only.
