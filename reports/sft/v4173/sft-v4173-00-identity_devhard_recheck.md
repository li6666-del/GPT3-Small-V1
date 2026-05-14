# sft-v4173-00-identity_devhard SFT Harness Report

- status: `failed`
- selected_step: `4`
- summary: Step 4 failed stage gates: identity_devhard_stage.

## Gate Results

- `PASS` v4173_no_mojibake [main/greedy]: 22/22 (1.00)
- `FAIL` identity_devhard_stage [stage/greedy]: 4/10 (0.40)
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
  output: Assistant发布时间: 2019-12-15。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'Assistant发布时间'
- prompt: 你的昵称是什么？
  output: 我不能保证与。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 别人该怎么称呼你？
  output: 我不能帮助称呼他人了。你可以叫我“姐姐".
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助', '姐姐'
- prompt: 请说明你的模型规模。
  output: 你可以查询norcania.
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'norcania'
- prompt: 你能说一下自己的来源吗？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助'
- prompt: 你的模型名字是什么？
  output: 我不能保证姓姓必提。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'

## Avoid Next Time

- 身份锚被侵蚀。下一轮应增加 identity anchor，并降低 unknown/ability 的相对权重。
