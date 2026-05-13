# V4.12 Round 04 Strategy Memo

- focus: `zh_factual_expand`
- mode: `stage_retry`
- init_checkpoint: `runs/sft-v412-02-concept_ml/step_000023.pt`
- learning_rate: `3e-07`
- reason: previous stage failed; retry same small target once: zh_factual_expand
- stage_rules: `simple_qa_week_days, simple_qa_boiling`

## Previous Round

- status: `failed`
- selected_step: `16`
- summary: Step 16 failed stage gates: simple_qa_boiling, simple_qa_week_days.

## Failure Memory Top Rules

- `unknown_no_certain_checkpoint`: 7
- `ability_simple_assistant`: 7
- `simple_qa_english_sky`: 7
- `unknown_no_fabrication_explicit`: 5
- `identity_not_chatgpt`: 1
- `simple_math_add`: 1
- `unknown_no_fabrication`: 1
- `simple_qa_boiling`: 1

## Decision

- Keep the round small: one focus, no English hard target, ability stays observe unless explicitly selected later.
- Preserve identity, stop, refusal, unknown safe, H2O, France, and 2+3 as main regression.
- Save checkpoint only if main gates and stage gates pass; otherwise delete `.pt` and keep report only.
