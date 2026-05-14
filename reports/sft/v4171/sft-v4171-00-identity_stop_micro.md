# sft-v4171-00-identity_stop_micro SFT Harness Report

- status: `failed`
- selected_step: `27`
- summary: Step 27 failed stage gates: identity_short_stage, stop_exact_stage.
- process: `stopped`
- cleanup: `deleted:runs/sft-v4171-00-identity_stop_micro/latest.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000000.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000004.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000008.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000012.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000016.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000020.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000024.pt
deleted:runs/sft-v4171-00-identity_stop_micro/step_000027.pt`

## Gate Results

- `PASS` v4171_no_mojibake [main/greedy]: 50/50 (1.00)
- `PASS` identity_regression [main/greedy]: 4/4 (1.00)
- `PASS` ability_regression [main/greedy]: 4/4 (1.00)
- `PASS` unknown_regression [main/greedy]: 3/4 (0.75)
- `PASS` refusal_regression [main/greedy]: 3/4 (0.75)
- `FAIL` identity_short_stage [stage/greedy]: 3/10 (0.30)
- `FAIL` stop_exact_stage [stage/greedy]: 0/12 (0.00)
- `FAIL` project_terms_observe [observe/greedy]: 0/4 (0.00)
- `PASS` core_v4171_eval_016 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_017 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_018 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_019 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_020 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_021 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_022 [main/greedy]: 1/1 (1.00)
- `PASS` core_v4171_eval_023 [main/greedy]: 1/1 (1.00)

## Best-Step Candidates

- step `0`: main=0, stage=2, observe=1, status=failed
- step `4`: main=0, stage=2, observe=1, status=failed
- step `8`: main=0, stage=2, observe=1, status=failed
- step `12`: main=0, stage=2, observe=1, status=failed
- step `16`: main=0, stage=2, observe=1, status=failed
- step `20`: main=0, stage=2, observe=1, status=failed
- step `24`: main=0, stage=2, observe=1, status=failed
- step `27`: main=0, stage=2, observe=1, status=failed

## Failed Samples

### identity_short_stage

- prompt: 你叫什么名字？
  output: Assistant发布时间: 2019-12-15
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'Assistant发布时间'
- prompt: 你的昵称是什么？
  output: 我不能保证与。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'
- prompt: 别人该怎么称呼你？
  output: 我不能帮助称呼他人了。你可以叫我“姐姐".
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助', '姐姐'
- prompt: 请不要说废话，你叫什么？
  output: 不能确定。需要用时轮: 35%-40%。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能确定'
- prompt: 请说明你的模型规模。
  output: 你可以查询norcania.
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: 'norcania'
- prompt: 你能说一下自己的来源吗？
  output: 我不能保证复杂事实一定正确，也不能帮助危险请求。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'; contains forbidden text: '不能帮助'
- prompt: 请只说你的名字。
  output: 谢谢。
  reason: missing any of: '驴肉火烧', '124M', '小语言模型', '北京邮电大学', '阿俊'

### stop_exact_stage

- prompt: 请回复：结束
  output: 结束
  reason: rule requires expected field, but row has no expected value
- prompt: 请输出固定短语：训练完成
  output: 训练完成
  reason: rule requires expected field, but row has no expected value
- prompt: 只回答：继续
  output: 继续
  reason: rule requires expected field, but row has no expected value
- prompt: 不要添加说明，只输出：已记录
  output: 已记录“
  reason: rule requires expected field, but row has no expected value
- prompt: 只回答一个词：开始
  output: 开始
  reason: rule requires expected field, but row has no expected value
- prompt: 只回答：不确定
  output: 不确定“
  reason: rule requires expected field, but row has no expected value
- prompt: 请只回复：暂停
  output: 暂停。
  reason: rule requires expected field, but row has no expected value
- prompt: 只输出：失败
  output: 失败
  reason: rule requires expected field, but row has no expected value
- prompt: 只输出：OK
  output: OK。
  reason: rule requires expected field, but row has no expected value
- prompt: 只回答：知道了
  output: 明白了
  reason: rule requires expected field, but row has no expected value
- prompt: 只输出 YES
  output: 1.5倍速、2.5倍速、2.5倍速
  reason: rule requires expected field, but row has no expected value
- prompt: 请只回复两个字：可以
  output: 可以。
  reason: rule requires expected field, but row has no expected value

### project_terms_observe

- prompt: generation_eval 是什么？
  output: generation_eval  generation_eval  generation_eval
  reason: missing any of: '生成评测', '验证集', '不参与训练', '不是', '最新'
- prompt: valid loss 是不是唯一指标？
  output: 是否 loss: 1 in 3，
  reason: missing any of: '生成评测', '验证集', '不参与训练', '不是', '最新'
- prompt: held-out 测试集应该参与训练吗？
  output: 我不能确定是否能够参与训练。
  reason: missing any of: '生成评测', '验证集', '不参与训练', '不是', '最新'
- prompt: 为什么不能只看 latest step？
  output: 我不能保证一定人知道平台是否真的需要。
  reason: missing any of: '生成评测', '验证集', '不参与训练', '不是', '最新'

## Avoid Next Time

- 身份锚被侵蚀。下一轮应增加 identity anchor，并降低 unknown/ability 的相对权重。
- 停止锚退化。下一轮应增加 exact stop 样本，并降低 max_new_tokens 干扰。
- 需要检查失败样本，避免只按 loss 决策。
