# V4.17.2 Strategy: Identity Short Only

上一轮复盘：V4.17.1 主线回归守住，但 identity_short 只有 3/10；stop exact 修正后为 5/12。

本轮改动：只主修身份短问法，stop exact 降为 observe。

原因：两个阶段目标一起修没有收益，身份短问法是更关键的助手外壳。

不修：broad QA、泛化算术、project_terms、stop exact。

保存标准：main regression 全过，identity_short_stage >= 0.70，且没有明显身份模板污染 ability/unknown。
