# sft-v4172-00-identity_short_only SFT Harness Report

- status: `failed`
- selected_step: `39`
- summary: Step 39 failed stage gates: identity_short_stage.
- process: `stopped`
- cleanup: `deleted:runs/sft-v4172-00-identity_short_only/latest.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000000.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000004.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000008.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000012.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000016.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000020.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000024.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000028.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000032.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000036.pt
deleted:runs/sft-v4172-00-identity_short_only/step_000039.pt`

## Gate Results

- `PASS` v4172_no_mojibake [main/greedy]: 27/27 (1.00)
- `FAIL` identity_short_stage [stage/greedy]: 5/10 (0.50)
- `PASS` identity_regression [main/greedy]: 2/2 (1.00)
- `PASS` ability_regression [main/greedy]: 2/2 (1.00)
- `PASS` unknown_regression [main/greedy]: 2/2 (1.00)
- `PASS` refusal_regression [main/greedy]: 2/2 (1.00)
- `PASS` stop_observe [observe/greedy]: 2/4 (0.50)
- `PASS` core_v4172_eval_018 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4172_eval_019 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4172_eval_020 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4172_eval_021 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4172_eval_022 [main/greedy]: 1/1 (1.00)

## Best-Step Candidates

- step `0`: main=0, stage=1, observe=0, status=failed
- step `4`: main=0, stage=1, observe=0, status=failed
- step `8`: main=0, stage=1, observe=0, status=failed
- step `12`: main=0, stage=1, observe=0, status=failed
- step `16`: main=0, stage=1, observe=0, status=failed
- step `20`: main=0, stage=1, observe=0, status=failed
- step `24`: main=0, stage=1, observe=0, status=failed
- step `28`: main=0, stage=1, observe=0, status=failed
- step `32`: main=0, stage=1, observe=0, status=failed
- step `36`: main=0, stage=1, observe=0, status=failed
- step `39`: main=0, stage=1, observe=0, status=failed

## Failed Samples

### identity_short_stage

- prompt: 你叫什么名字？
  output: Assistant发布时间: 2019-12-15
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'Assistant发布时间'
- prompt: 请不要说废话，你叫什么？
  output: Icelandic。 Noted.
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 请说明你的模型规模。
  output: 你可以查询norcania.
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'norcania'
- prompt: 你能说一下自己的来源吗？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助'
- prompt: 请只说你的名字。
  output: A.是模型精。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'

## Avoid Next Time

- 身份锚被侵蚀。下一轮应增加 identity anchor，并降低 unknown/ability 的相对权重。
