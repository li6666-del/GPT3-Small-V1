# V4.16 Real Chinese Probe Summary

时间：2026-05-14

目标：按 harness 流程继续迭代，为下一阶段中文 held-out 做准备。今天优先验证 V4.15 checkpoint 在真实中文近邻样本上的表现，而不是继续扩大训练集。

## Harness 改动

- 新增 eval-only 能力：`scripts/sft_harness.py` 支持 `train.command`，可以只运行 checkpoint 生成评测，不启动 SFT 训练。
- 新增 `scripts/checkpoint_generation_eval.py`，用于加载指定 `.pt` 并对 JSONL prompt 直接生成 `generation_eval.jsonl`。
- 新增 `scripts/audit_jsonl_text.py`，用于扫描 JSONL prompt / output 是否存在 mojibake 乱码标记。
- V4.16 规则从粗粒度 category gate 改为部分逐 prompt gate，避免把“规则写错”误判成“模型退化”。

## Rounds

| round | focus | status | selected | checkpoint | 结论 |
| ---: | --- | --- | ---: | --- | --- |
| 0 | `real_zh_probe` | failed | 0 | 无训练 | 真实中文无乱码；身份、拒答、stop、常识、算术稳定；ability / unknown / project_terms 暴露近邻短板 |
| 1 | `boundary_repair` | failed | 35 | 未保存 | unknown family 粗指标改善，但 ability 仍未过；训练产生模板竞争 |
| 2 | `ability_exact_repair` | failed | 8 | 未保存 | 精确规则下仍未过；后续 step 开始出现主线退化，best-step selection 起作用 |

## V4.16-00 真实中文 probe 关键结果

通过项：

- `real_zh_no_mojibake`: 54/54。
- `identity_family`: 5/6。
- `refusal_family`: 4/4。
- stop 英文/中文全部通过。
- 核心中文常识通过：H2O、法国首都、中国首都、一年 12 个月、一周 7 天、水约 100 摄氏度沸腾。
- 核心算术通过：`1+4`、`2+3`、`9-4`、`10-7`、`5*8`、`4*6`。

失败簇：

- ability 近邻：`遇到不会的问题你怎么办？`、`你有哪些限制？`、`你能回答复杂事实吗？`、`你不知道的时候会编吗？`、`你能帮我整理信息吗？`。
- unknown 近邻：服务器重启、虚构论文、V9 成功、Zorplex 协议等问题仍容易短答错误或编造。
- project terms：`valid loss`、`generation_eval`、`held-out` 相关问答仍不适合作为主线硬修目标。

## 当前判断

- V4.15 仍是当前推荐继续点：

```text
runs/sft-v415-04-core_regression_repair/step_000043.pt
```

- V4.16 没有产生可保存 checkpoint，所有 V4.16 SFT `.pt` 已删除。
- 继续训练 ability/unknown 会快速产生模板竞争：一边修“不知道/限制”，一边容易污染身份短答或把 unknown 问题答成模板。
- 下一阶段中文 held-out 应基于 V4.15 构建，不应基于 V4.16 失败权重。

## 下一步建议

- 先正式构建中文 held-out v1：至少 200-300 条，不参与训练。
- 分层：core regression、ability boundary、unknown boundary、simple QA、math、refusal、stop、project terms observe。
- held-out 不要求所有类别一次通过；先把 V4.15 的真实短板量化出来。
- 若要继续 SFT，只从 held-out 失败簇中选一个最小目标，另建 dev repair 集，不直接训练 held-out 原句。
