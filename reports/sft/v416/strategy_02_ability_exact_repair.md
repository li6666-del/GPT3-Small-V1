# V4.16-02 Strategy: Exact Ability Boundary Repair

目标：修 V4.16-00/01 暴露的 5 个真实中文能力边界问法。

策略：允许使用 probe 中的失败问法作为 dev repair，不把它们作为未来正式 held-out。

防护：加入 Zorplex、虚构论文、V9、checkpoint 等 unknown 锚，防止 ability 修复污染未知边界。
