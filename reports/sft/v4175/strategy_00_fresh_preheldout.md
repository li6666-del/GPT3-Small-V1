# V4.17.5 Strategy: Fresh Pre-Heldout Eval

目标：不训练，评测 V4.17.4 checkpoint 是否值得进入更大 held-out。

checkpoint：runs/sft-v4174-00-identity_template_redirect/step_000008.pt

重点：fresh identity、ability、unknown、refusal、semantic stop、核心 QA/math。

不把 broad QA、泛化算术、project_terms 作为本轮硬门槛。
