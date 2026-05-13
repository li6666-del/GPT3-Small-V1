# V4.15 Ability Must-Fix Summary

目标：能力说明必须修好，同时守住身份、stop、拒答、unknown、一周七天、`1+4`、基础减法和沸腾点。

## Rounds

| round | focus | status | accepted | selected | init_after | summary |
| ---: | --- | --- | --- | ---: | --- | --- |
| 0 | `ability_acquire_balanced` | failed | false | 63 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | `你能做什么？` 已改善，但 `你可以帮我做什么？` 未过。 |
| 1 | `canhelp_retry` | failed | false | 0 | `runs/sft-v414-00-short_qa_corrections/step_000036.pt` | 后期 step 修好 ability 家族，但 `1+4` 退化；保留临时 step 48。 |
| 2 | `anchor_repair` | failed | false | 55 | `runs/sft-v415-01-canhelp_retry/step_000048.pt` | ability 全过，但 `1+4` 仍未修回。 |
| 3 | `math_repair_strong` | passed | true | 51 | `runs/sft-v415-03-math_repair_strong/step_000051.pt` | ability 与 `1+4` 同时通过，但抽查发现 `9-4` 和沸腾点退化。 |
| 4 | `core_regression_repair` | passed | true | 43 | `runs/sft-v415-04-core_regression_repair/step_000043.pt` | 严格 core regression 通过，作为 V4.15 最终点。 |

## Final Checkpoint

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
```

## Passed Main Gates

- 身份：`你是谁？`
- 非 ChatGPT：`你是 ChatGPT 吗？`
- stop：英文 `Done.` 和中文 `完成`
- 安全拒答：盗取密码、绕过验证、入侵邮箱
- unknown：不确定不编造，checkpoint 不确定性
- 中文常识：H2O、法国首都、中国首都、一年 12 个月、一周 7 天、水约 100 摄氏度沸腾
- 算术：`1+4`、`2+3`、`9-4`、`10-7`、`5*8`、`4*6`
- 能力说明：`你能做什么？`、`你的能力是什么？`、`你可以帮我做什么？`、`请简短说明你的能力。`

## Remaining Issue

- 英文 sky 仍失败，输出 `Average day: Average day:`。继续作为 observe，不进入中文主线。

## Judgment

V4.15 达成目标：能力说明已从 observe/失败项升级为可通过硬门槛的能力。当前实验候选继续点更新为 V4.15 step 43。下一步 V4.16 应扩大中文 held-out，验证这些修复是否有近邻泛化，而不是继续堆同类训练样本。
