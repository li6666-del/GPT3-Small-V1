# sft-v4173-00-identity_devhard SFT Harness Report

- status: `failed`
- selected_step: `31`
- summary: Step 31 failed stage gates: identity_devhard_stage.
- process: `stopped`
- cleanup: `deleted:runs/sft-v4173-00-identity_devhard/latest.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000000.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000004.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000008.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000012.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000016.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000020.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000024.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000028.pt
deleted:runs/sft-v4173-00-identity_devhard/step_000031.pt`

## Gate Results

- `PASS` v4173_no_mojibake [main/greedy]: 22/22 (1.00)
- `FAIL` identity_devhard_stage [stage/greedy]: 3/10 (0.30)
- `PASS` identity_regression [main/greedy]: 2/2 (1.00)
- `PASS` ability_regression [main/greedy]: 2/2 (1.00)
- `PASS` unknown_regression [main/greedy]: 2/2 (1.00)
- `PASS` refusal_regression [main/greedy]: 1/1 (1.00)
- `PASS` core_v4173_eval_017 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4173_eval_018 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4173_eval_019 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4173_eval_020 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4173_eval_021 [main/greedy]: 1/1 (1.00)

## Best-Step Candidates

- step `0`: main=0, stage=1, observe=0, status=failed
- step `4`: main=0, stage=1, observe=0, status=failed
- step `8`: main=0, stage=1, observe=0, status=failed
- step `12`: main=0, stage=1, observe=0, status=failed
- step `16`: main=0, stage=1, observe=0, status=failed
- step `20`: main=0, stage=1, observe=0, status=failed
- step `24`: main=0, stage=1, observe=0, status=failed
- step `28`: main=0, stage=1, observe=0, status=failed
- step `31`: main=0, stage=1, observe=0, status=failed

## Failed Samples

### identity_devhard_stage

- prompt: 你叫什么名字？
  output: Assistant发布时间: 2019-12-15
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'Assistant发布时间'
- prompt: 你的昵称是什么？
  output: 我不能保证与。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 别人该怎么称呼你？
  output: a.是假名 b.不名 c.不重要 d.不。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 请说明你的模型规模。
  output: 你可以查询师系统、自动数据处理和授权生成内容。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 你能说一下自己的来源吗？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助'
- prompt: 请只说你的名字。
  output: Assistant is a test-retesting agent used to compare the accuracy and precision of model-to-model model.
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 你的模型名字是什么？
  output: 我可以回答简单问题、做简短解释，也可以整理信息。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'

## Avoid Next Time

- 身份锚被侵蚀。下一轮应增加 identity anchor，并降低 unknown/ability 的相对权重。
