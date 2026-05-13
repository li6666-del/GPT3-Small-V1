# V4.11 Ten-Round Micro-Loop Summary

时间：2026-05-13

目标：在 SFT Harness 上连续自迭代十轮，验证 best-step selection、分层 gates、失败自动清理和小步 SFT 策略，并向“简单问答助手”推进。

## 自动化流程改动

- `eval_sft_outputs.py` 支持 `step: best_complete`，不再只看最新 step。
- gates 分为 `main`、`stage`、`observe`：
  - `main`：身份、stop、refusal、基础 unknown，不允许退化。
  - `stage`：当前轮专门修什么，就检查什么。
  - `observe`：记录但不一票否决。
- 通过的轮次只保留选中的 best checkpoint。
- 失败的轮次删除该 run 下全部 `.pt`，只保留日志和报告。

## 十轮结果

| round | variant | status | selected step | 结论 |
| ---: | --- | --- | ---: | --- |
| 0 | ability | failed | 29 | 旧 gate 把 unknown checkpoint 作为 main，失败；无权重保留 |
| 1 | nofab | failed | 15 | 旧 gate 下 unknown checkpoint 失败；无权重保留 |
| 2 | english_sky | failed | 25 | 英文 sky 未学会；无权重保留 |
| 3 | zh_qa | passed | 29 | 主门槛通过，保留 checkpoint |
| 4 | math | passed | 29 | 主门槛通过，接续 V4.11-03 并补强 `2+3`，保留 checkpoint |
| 5 | ability_nofab | failed | 29 | ability、nofab explicit、unknown checkpoint 阶段门槛失败 |
| 6 | english_zh | failed | 5 | 英文 sky 阶段门槛失败 |
| 7 | core_mix | failed | 5 | 四个阶段门槛同时失败，混合任务干扰明显 |
| 8 | core_mix_low | failed | 25 | 低学习率仍未解决核心阶段门槛 |
| 9 | final_core | failed | 29 | 混合 core 仍失败，停止 |

## 云端保留权重

```text
runs/sft-v411-03-zh_qa-micro/step_000029.pt
runs/sft-v411-04-math-micro/step_000029.pt
```

其它 V4.11 `.pt` 已清理。训练结束后云端无残留 SFT 进程。

## 主要发现

- V4.11-03 和 V4.11-04 说明小步局部 SFT 有收益：中文简单问答和 `2+3` 可以在不破坏主门槛的前提下被补强。
- ability prompt 仍会吸到异常表达，例如“能做决断...”一类残留模式。它不是靠混合数据顺手修好的问题。
- `unknown_no_fabrication_explicit` 多数是语义接近但词面不过关，后续要么改成语义规则，要么专门训练“不能编造”措辞。
- 英文简单问答仍受 `Average...` 旧模式污染，短期不适合和中文助手主线混训。
- core mix 会放大干扰。把 ability、英文、unknown explicit 混在同一轮，失败概率高。

## 当前判断

V4.7.1 仍是保守主线基线。V4.11-04 是当前最有价值的实验候选继续点，但还不应直接宣布为正式主线。

下一步建议：

- 先用 V4.11-04 和 V4.7.1 做一组固定回归对比，确认中文问答和数学收益是否稳定。
- ability 单独开极小实验，只训练 `你能做什么？/你有什么限制？` 这一族 prompt。
- 英文 simple QA 暂时独立成英文课程，不和中文助手主线混合。
- unknown explicit 先改评测规则或单独做措辞课程，不再把它混进 core mix。

## Harness v0.2 约束

V4.11 暴露出一个流程问题：预设十轮虽然能自动执行，但“下一轮策略”不应提前写死。后续 harness 工作流必须加入三条硬约束：

1. 每轮完成后，下一轮开始前，必须读取上一轮报告和历史 `failure_memory.jsonl`，明确改变下一轮策略。策略变化至少包括：主攻目标、放弃或降级的指标、数据比例、gate 分层、起始 checkpoint。
2. 每轮步子必须小。默认只主修一个能力，最多一个辅助能力；其它能力只做 main regression 或 observe，避免 core mix 干扰。
3. checkpoint 由实验结果自动筛选，但最终由 Codex 判断是否有未来价值。保存标准不是“训练跑完”，而是：
   - main gates 不退化；
   - 当前 stage 目标有明确收益；
   - 相比起点没有引入更大的行为污染；
   - 未来实验可以把它作为合理起点。

因此，下一阶段不再使用“预设多轮固定变体”作为主流程。正确流程是：

```text
上一轮 report + failure_memory
-> Codex 形成下一轮 strategy memo
-> 生成小步数据和 YAML
-> 训练、评测、清理
-> 保存或删除 checkpoint
-> 写入报告和失败经验
-> 再决定下一轮
```
