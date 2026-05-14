# V4.17-00 Strategy: Chinese Held-out v1

目标：冻结 V4.15，建立正式中文 held-out v1 基线。

原则：本轮只评测不训练；held-out 原句后续不得进入训练集。

类别：identity、ability、unknown、refusal、stop、qa、math、project_terms。

决策：根据失败簇选择下一轮一个最小 dev repair 目标。
