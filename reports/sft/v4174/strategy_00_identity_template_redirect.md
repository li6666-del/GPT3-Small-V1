# V4.17.4 Strategy: Identity Template Redirect

上一轮复盘：短答名字形式不稳，dev-hard 强压后没有形成稳定泛化。

本轮改动：不再强求短答名字，所有名字/昵称/来源/规模问法统一导向已稳定的完整身份模板。

理由：扬长避短，利用模型已学会的完整身份回答，而不是继续教难以稳定的短输出格式。

主修：身份模板重定向。

辅助：ability/unknown/refusal/QA/math/stop 核心回归。

不修：broad QA、泛化算术、project_terms、strict stop exact。

保存标准：main regression 全过，identity_template_stage >= 0.80。
