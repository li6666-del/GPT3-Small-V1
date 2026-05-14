# V4.17.1 Strategy: Identity Short + Stop Exact Micro

上轮依据：V4.17 held-out 显示 refusal/stop 语义是强项，identity 短问法和 stop exact 是低风险高收益短板。

主修：身份短问法、模型昵称/称呼、真正 stop exact 失败。

辅助：只保留 refusal / unknown / ability / 核心 QA / 已有算术锚作为 regression。

明确不修：broad QA、泛化算术、project_terms。它们继续 observe，避免污染主线。

数据原则：不训练 V4.17 held-out 原题，只使用近邻改写和核心回归锚。

保存标准：main regression 不退化，identity_short_stage >= 0.80，stop_exact_stage >= 0.75。
