# V4.16-01 Strategy: Boundary Repair

目标：只修真实中文 probe 暴露的 ability/unknown 近邻失败。

训练原则：不用 V4.16-00 probe 原句；只用近邻改写和核心锚点。

保留强项：身份、拒答、stop、中文常识、基础算术全部作为 regression gate。

不修项目术语：project_terms 继续 observe，避免引入半中半英污染。
