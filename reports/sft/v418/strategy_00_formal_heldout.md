# V4.18 Strategy: Formal Held-out

目标：不训练，正式评测 V4.17.8 当前候选 checkpoint。

checkpoint：runs/sft-v4178-00-preheldout_mainline_gate/step_000031.pt

分层：

- main：identity、refusal、unknown、stop semantic、核心 QA/math anchors。
- stage：identity/unknown near-neighbor。
- observe：ability fresh、project terms、broad QA、strict stop exact、泛化算术。

注意：V4.17 已进入 dev-hard 的样本不作为本轮正式 held-out 原题。
