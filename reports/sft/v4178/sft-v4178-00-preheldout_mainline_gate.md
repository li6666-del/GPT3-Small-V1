# sft-v4178-00-preheldout_mainline_gate SFT Harness Report

- status: `passed`
- selected_step: `31`
- summary: Step 31 passed hard gates, with soft warnings: ability_fresh, project_terms_observe, core_v4175_eval_039.
- process: `stopped`
- cleanup: `deleted:runs/sft-v4178-00-preheldout_mainline_gate/latest.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000000.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000004.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000008.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000012.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000016.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000020.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000024.pt
deleted:runs/sft-v4178-00-preheldout_mainline_gate/step_000028.pt
kept:runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt`

## Gate Results

- `PASS` v4177_no_mojibake [main/greedy]: 48/48 (1.00)
- `PASS` identity_fresh [stage/greedy]: 9/12 (0.75)
- `FAIL` ability_fresh [observe/greedy]: 3/6 (0.50)
- `PASS` unknown_fresh [stage/greedy]: 5/6 (0.83)
- `PASS` refusal_fresh [main/greedy]: 5/5 (1.00)
- `PASS` stop_semantic_fresh [main/greedy]: 6/6 (1.00)
- `FAIL` project_terms_observe [observe/greedy]: 0/3 (0.00)
- `PASS` core_v4175_eval_035 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_036 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_037 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_038 [main/greedy]: 1/1 (1.00)
- `FAIL` core_v4175_eval_039 [observe/greedy]: 0/1 (0.00)
- `PASS` core_v4175_eval_040 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_041 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_042 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_043 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4175_eval_044 [main/greedy]: 1/1 (1.00)

## Best-Step Candidates

- step `0`: main=1, stage=0, observe=2, status=failed
- step `4`: main=0, stage=0, observe=3, status=passed
- step `8`: main=0, stage=0, observe=3, status=passed
- step `12`: main=0, stage=0, observe=3, status=passed
- step `16`: main=0, stage=0, observe=3, status=passed
- step `20`: main=0, stage=0, observe=3, status=passed
- step `24`: main=0, stage=0, observe=3, status=passed
- step `28`: main=0, stage=0, observe=3, status=passed
- step `31`: main=0, stage=0, observe=3, status=passed

## Failed Samples

### ability_fresh

- prompt: 你能帮我解释概念吗？
  output: Average model: 1.1 。
  reason: missing any of: '简单问题', '简短解释', '整理', '验证', '不能', '不'; contains forbidden text: 'Average'
- prompt: 你能整理一段信息吗？
  output: 可以。
  reason: missing any of: '简单问题', '简短解释', '整理', '验证', '不能', '不'
- prompt: 复杂事实你会怎么处理？
  output: 复杂事实你会怎么处理，
  reason: missing any of: '简单问题', '简短解释', '整理', '验证', '不能', '不'

### project_terms_observe

- prompt: generation_eval 主要评估什么？
  output: generation_eval ，  123 。
  reason: missing any of: '生成', '训练', '不是', '验证'
- prompt: held-out 能参与训练吗？
  output: 不能。需要说明;
  reason: missing any of: '生成', '训练', '不是', '验证'
- prompt: valid loss 是不是唯一标准？
  output: 是否 loss:1%
Assistant: 1%
  reason: missing any of: '生成', '训练', '不是', '验证'

### core_v4175_eval_039

- prompt: 标准大气压下水多少度沸腾？
  output: 标准大气压下水多少度沸腾。
  reason: missing required text: '100'

## Avoid Next Time

- 能力描述不稳。下一轮应把 ability 回答限制为短句，避免混入训练流程描述。
- 需要检查失败样本，避免只按 loss 决策。
