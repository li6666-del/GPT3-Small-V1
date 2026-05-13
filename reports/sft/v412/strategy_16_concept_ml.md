# V4.12 Round 16 Strategy Memo

- focus: `concept_ml`
- mode: `normal`
- init_checkpoint: `runs/sft-v412-15-math_add/step_000000.pt`
- learning_rate: `4e-07`
- reason: previous checkpoint accepted; advance to next small target
- stage_rules: `concept_machine_learning, concept_overfit`

## Previous Round

- status: `passed`
- selected_step: `0`
- summary: Step 0 passed hard gates, with soft warnings: ability_simple_assistant.

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
