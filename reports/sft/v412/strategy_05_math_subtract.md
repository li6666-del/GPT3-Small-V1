# V4.12 Round 05 Strategy Memo

- focus: `math_subtract`
- mode: `skip_failed_target`
- init_checkpoint: `runs/sft-v412-02-concept_ml/step_000023.pt`
- learning_rate: `5e-07`
- reason: previous target failed after retry budget; move to next small target
- stage_rules: `math_sub_9_4, math_sub_10_7`

## Previous Round

- status: `failed`
- selected_step: `16`
- summary: Step 16 failed stage gates: simple_qa_boiling, simple_qa_week_days.

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
