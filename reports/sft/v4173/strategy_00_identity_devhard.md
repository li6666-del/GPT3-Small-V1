# V4.17.3 Strategy: Identity Dev-Hard

上一轮复盘：V4.17.2 identity_short 从 3/10 提升到 5/10，但仍未过保存线。

本轮改动：把 V4.17 诊断中暴露的身份短问法转为 dev-hard 训练目标。

代价：V4.17 这批身份短问法不再作为最终 held-out，后续必须重新生成 fresh held-out。

主修：身份短问法。

辅助：仅保留 ability/unknown/refusal/QA/math/stop 核心回归。

不修：broad QA、泛化算术、project_terms、stop exact。

保存标准：main regression 全过，identity_devhard_stage >= 0.80。
