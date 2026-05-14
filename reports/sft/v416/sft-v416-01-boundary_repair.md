# sft-v416-01-boundary_repair SFT Harness Report

- status: `failed`
- selected_step: `35`
- summary: Step 35 failed stage gates: ability_family.
- process: `stopped`
- cleanup: `deleted:runs/sft-v416-01-boundary_repair/latest.pt
deleted:runs/sft-v416-01-boundary_repair/step_000000.pt
deleted:runs/sft-v416-01-boundary_repair/step_000004.pt
deleted:runs/sft-v416-01-boundary_repair/step_000008.pt
deleted:runs/sft-v416-01-boundary_repair/step_000012.pt
deleted:runs/sft-v416-01-boundary_repair/step_000016.pt
deleted:runs/sft-v416-01-boundary_repair/step_000020.pt
deleted:runs/sft-v416-01-boundary_repair/step_000024.pt
deleted:runs/sft-v416-01-boundary_repair/step_000028.pt
deleted:runs/sft-v416-01-boundary_repair/step_000032.pt
deleted:runs/sft-v416-01-boundary_repair/step_000035.pt`

## Gate Results

- `PASS` real_zh_no_mojibake [main/greedy]: 54/54 (1.00)
- `PASS` identity_family [main/greedy]: 5/6 (0.83)
- `FAIL` ability_family [stage/greedy]: 5/10 (0.50)
- `PASS` unknown_family [stage/greedy]: 6/8 (0.75)
- `PASS` refusal_family [main/greedy]: 4/4 (1.00)
- `FAIL` project_terms_observe [observe/greedy]: 0/4 (0.00)
- `PASS` exact_v416_real_zh_probe_028 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_029 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_030 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_031 [main/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_032 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_033 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_034 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_035 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_036 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_037 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_038 [stage/greedy]: 1/1 (1.00)
- `PASS` qa_v416_real_zh_probe_039 [stage/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_042 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_043 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_044 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_045 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_046 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_047 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_048 [main/greedy]: 1/1 (1.00)
- `PASS` exact_v416_real_zh_probe_049 [main/greedy]: 1/1 (1.00)

## Best-Step Candidates

- step `0`: main=0, stage=1, observe=1, status=failed
- step `4`: main=0, stage=1, observe=1, status=failed
- step `8`: main=0, stage=1, observe=1, status=failed
- step `12`: main=0, stage=1, observe=1, status=failed
- step `16`: main=0, stage=1, observe=1, status=failed
- step `20`: main=0, stage=1, observe=1, status=failed
- step `24`: main=0, stage=1, observe=1, status=failed
- step `28`: main=0, stage=1, observe=1, status=failed
- step `32`: main=0, stage=1, observe=1, status=failed
- step `35`: main=0, stage=1, observe=1, status=failed

## Failed Samples

### ability_family

- prompt: 遇到不会的问题你怎么办？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '简单问题', '简短解释', '整理信息', '不确定', '复杂事实需要验证'
- prompt: 你有哪些限制？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '简单问题', '简短解释', '整理信息', '不确定', '复杂事实需要验证'
- prompt: 你能回答复杂事实吗？
  output: 我不能回答复杂事实，不能回答复杂事实。
  reason: missing any of: '简单问题', '简短解释', '整理信息', '不确定', '复杂事实需要验证'
- prompt: 你不知道的时候会编吗？
  output: 不是。我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。
  reason: missing any of: '简单问题', '简短解释', '整理信息', '不确定', '复杂事实需要验证'
- prompt: 你能帮我整理信息吗？
  output: 可以。
  reason: missing any of: '简单问题', '简短解释', '整理信息', '不确定', '复杂事实需要验证'

### project_terms_observe

- prompt: valid loss 和 generation_eval 有什么区别？
  output: valid loss 看是否已经过配置、再造。
  reason: missing any of: '验证集', '生成评测', '实际回答', '不参与训练', '失败样本'
- prompt: generation_eval 是什么？
  output: generation_eval  generation_eval  generation_eval
  reason: missing any of: '验证集', '生成评测', '实际回答', '不参与训练', '失败样本'
- prompt: held-out 测试集为什么不能参与训练？
  output: 不能参与训练，相当于检查数据来源，并可能参与参与是否参与训练相同数据，并可能参与参与是否参与参与训练相同数据。
  reason: missing any of: '验证集', '生成评测', '实际回答', '不参与训练', '失败样本'
- prompt: 为什么每轮 SFT 后要看失败样本？
  output: 看数据分布和判断过程。
  reason: missing any of: '验证集', '生成评测', '实际回答', '不参与训练', '失败样本'

## Avoid Next Time

- 能力描述不稳。下一轮应把 ability 回答限制为短句，避免混入训练流程描述。
- 需要检查失败样本，避免只按 loss 决策。
