# 项目代码审查报告

- 项目：GPT3-Small-V1
- 路径：`D:\GPT3 small V1`
- 日期：2026-05-15
- 方法：主持人专家多智能体流
- 范围：`gpt_small/`、`scripts/`、`configs/`、`experiments/`、`reports/`、项目文档

## 主持人黑板

目标：从功能性、可靠性、安全性、简洁性四个角度审查项目代码，输出可执行的风险报告。

约束：

- 本轮只审查，不修改业务代码。
- 保留 SFT Harness 迭代规则作为审查准则：每轮训练前应复盘上一轮 report 和 `reports/sft/failure_memory.jsonl`，并保持小步迭代。
- 未跟踪文件 `scripts/chat_checkpoint.py` 纳入审查，但不修改。

角色：

- 功能性/正确性专家：训练、SFT、生成评测、checkpoint 判断链路。
- 可靠性/性能专家：失败模式、恢复、远程训练、日志、资源波动。
- 安全性专家：命令执行、SSH、凭据、反序列化、下载供应链。
- 简洁性/可维护性专家：复现链条、脚本重复、配置膨胀、文档同步。
- 批判者：复核排序、证据强度和误报风险。

验证记录：

- 已运行：`python -m compileall -f gpt_small scripts`
- 结果：通过，所有 Python 文件语法编译成功。
- 限制：未运行训练、远程 harness 或真实 checkpoint 评测；本报告基于静态审查和轻量编译验证。

## 总体结论

项目核心模型实现相对清晰，训练、SFT、生成评测和 harness 已形成完整工程闭环。最高风险不在 Transformer 本体，而在 SFT 迭代判断链路：generation eval 产物可能被旧结果、重复行或未完成写入污染；prompts 缺失时完整性判断会退化；这些问题会直接影响 best-step、hard gate 和 checkpoint 去留。

第二类高风险是可复现性和流程合规：文档声明 V4.18.9 是当前最佳，但仓库里缺少完整可追踪的 V4.18.9 及后续轮次材料；harness 也没有把“先读上一轮 report 和 failure memory”做成启动训练前的硬约束。第三类风险是安全边界：SSH 自动信任主机密钥，以及 YAML 作为本地/远程执行计划时权限较大。

建议优先顺序：

1. 修复评测产物完整性：append 污染、prompts 缺失、轮询读取未完成文件、重复 prompt id。
2. 补齐 V4.18.9 及后续轮次的策略、YAML、报告和小型元数据，或者在文档中明确不可完整复现。
3. 把“复盘上一轮 report + failure memory + strategy memo”做成 harness 硬约束。
4. 收紧 SSH 主机校验和 YAML 命令执行面。
5. 改进 checkpoint 原子保存、反序列化安全、远程日志、脚本复用和配置结构。

## 主要发现

### 1. 高危：评测产物完整性不足，可能污染 gate 和 checkpoint 去留

位置：

- `gpt_small/training/sft.py:132`
- `gpt_small/training/sft.py:253`
- `scripts/sft_harness.py:163`
- `scripts/sft_harness.py:264`
- `scripts/eval_sft_outputs.py:42`
- `scripts/eval_sft_outputs.py:173`
- `scripts/eval_sft_outputs.py:178`

证据：

- `run_generation_eval()` 使用 append 模式写 `generation_eval.jsonl`。
- harness 只有在 YAML 设置 `clear_run_dir=true` 时才清理旧产物。
- `count_jsonl()` 在 prompts 路径缺失时返回 `None`。
- `complete_steps()` 在 `expected_prompts is None` 时，只要每个 mode 有至少一行就认为 step complete。
- harness 轮询下载同一个 JSONL 文件后直接评估，没有临时文件、rename、manifest 或 step done 标记。
- 评测完整性统计按 `(step, mode)` 计数，不按 `(step, mode, id)` 去重。

触发场景：

- 重跑同一个 `out_dir`，旧 `generation_eval.jsonl` 未删除。
- 远端训练正在写 generation eval，harness 恰好下载到半写入文件。
- `evaluation.prompts_path` 写错、本地未生成 eval prompts、路径相对 `local_root` 解析失败。
- generation eval 重复写入同一个 prompt id。

影响：

- 旧 step 或重复 row 可能被当成当前完整评测。
- 未完整写完的 step 可能提前进入 gate。
- `best_complete`、hard gate pass/fail、checkpoint 保留/删除都可能基于脏数据。

主持人裁决：保留为最高优先级。它直接影响本项目最核心的 checkpoint 判断链路。

建议：

- SFT 启动时，在 `resume=false` 或 `fresh_generation_eval=true` 时删除旧 `generation_eval.jsonl`。
- generation eval 先写临时文件或 step 分片，完成后原子 rename，或写 manifest/done 标记。
- harness 只评估带完成标记的 step。
- prompts 配置存在但文件缺失/为空时 fail fast 或返回 `incomplete`，不要静默降级到 `expected_prompts=None`。
- 评测端按 `(step, mode, id)` 去重；重复 id 应报告为数据污染。
- 报告写入 generation eval 文件行数、唯一 prompt 数、重复数、每个 step/mode 覆盖数。

### 2. 高危：文档声明的 V4.18.9 主线缺少完整可复现链条

位置：

- `README.md:11`
- `README.md:66-96`
- `项目进程.md:98`
- `.gitignore:11`
- `reports/sft/failure_memory.jsonl`

证据：

- README 和项目进程文档声明当前最佳为 V4.18.9，并记录 V4.18.10-V4.18.12 结论。
- 当前 `reports/sft/` 只保留 `failure_memory.jsonl`，没有 V4.18.9 markdown report。
- 仓库可追踪脚本主要到 `scripts/run_v418_formal_heldout_eval.py`，未看到完整的 v4189 源脚本、YAML、config 和最终报告。
- 大型产物被忽略是合理的，但小型策略和报告元数据也缺失，导致结论难以审计。

触发场景：新维护者或未来自己尝试复现“V4.18.9 为什么被选为当前最佳 checkpoint”。

影响：

- 无法从当前仓库独立复现 V4.18.9 的选择依据。
- 难以审计 V4.18.10-V4.18.12 为什么未保留。
- 后续迭代容易从错误起点或不完整经验继续。

主持人裁决：保留为高优先级。它是结论可信度和工程可审计性问题，不只是文档瑕疵。

建议：

- 每轮至少保留：策略 memo、数据生成脚本、实验 YAML、最终 report、小型 eval rules/manifest。
- 大文件继续忽略，必要时只记录 manifest/hash 和外部路径。
- 若某 checkpoint 只是本地快照，README 应明确“当前仓库不可完整复现该轮”。
- README 常用命令应标注 `v4178` 是历史示例，或更新为最新可复现入口。

### 3. 高危：harness 没有强制执行“先复盘上一轮 report 和 failure memory”

位置：`scripts/sft_harness.py:348-357`

证据：`run_once()` 加载实验后立即 build、连接远端、上传、启动训练；没有读取上一轮 markdown report，也没有验证 `reports/sft/failure_memory.jsonl` 已被复盘。

触发场景：直接运行任意 experiment YAML 或新增轮次脚本。

影响：

- AGENTS.md 中的 SFT Harness 迭代规则只能靠人工遵守。
- 容易重复训练历史失败方向。
- 无法强制说明为什么改策略、主修什么、降级什么、起点 checkpoint 选哪个。

主持人裁决：保留为高优先级流程合规缺口。

建议：

- 在 experiment YAML 增加 `strategy.previous_report`、`strategy.failure_memory`、`strategy.memo_path`。
- harness 启动训练前读取并校验这些文件。
- strategy memo 应明确：改策略原因、主修能力、辅助能力、放弃/降级项、起点 checkpoint。
- 缺失上一轮材料或 strategy memo 时 fail fast。

### 4. 高危：SSH 自动信任主机密钥，可能泄露远程密码

位置：

- `scripts/sft_harness.py:70-76`
- `scripts/upload_cache_to_autodl.py:59-65`

证据：Paramiko 使用 `AutoAddPolicy()`，首次连接或主机变化时自动接受主机密钥。

触发场景：DNS/网络被劫持，或首次连接到伪造远端主机。

影响：`AUTODL_PASSWORD` 可能被中间人获取；训练数据、checkpoint、远程命令均可能被劫持或篡改。

主持人裁决：保留为安全类最高优先级。项目依赖远程 GPU 执行，远程身份校验是基础安全边界。

建议：

- 改用 `RejectPolicy`，加载固定 `known_hosts`。
- 支持显式主机指纹配置。
- 优先使用 SSH key，减少密码认证暴露面。
- 如果曾在不可信网络首次连接远端，建议轮换 AutoDL 密码。

### 5. 中高：YAML 命令执行面过大，本地和远程都有误操作/执行风险

位置：

- `scripts/sft_harness.py:132-137`
- `scripts/sft_harness.py:170-190`
- `scripts/sft_harness.py:302-342`

证据：

- `data.build_command` 直接进入 `subprocess.run(..., shell=True)`。
- `train.command` 可从 YAML 自定义，并进入远程 `bash -lc`。
- cleanup 根据配置拼接远程清理 `.pt` 的 shell 片段。

触发场景：运行外部、下载得到、被篡改或误编辑的 experiment YAML。

影响：

- 本地可执行任意 shell 命令，读取或外传 `AUTODL_PASSWORD`、SSH key、仓库数据、checkpoint。
- 远程可执行任意训练命令；默认远程用户常为 `root` 时破坏半径更大。
- cleanup 可能删除非预期目录中的 `.pt` 文件。

主持人裁决：保留为中高优先级。若 YAML 始终是可信内部文件，它更像危险执行面和误操作风险；一旦 YAML 进入共享/下载/自动生成链路，就升级为可利用风险。

建议：

- 将 `build_command` 改成参数数组，使用 `shell=False`。
- 只允许固定白名单脚本或固定任务类型。
- 禁用任意字符串形式的远程 `train.command`，改为模板化命令和受控参数。
- 校验 `run_dir` 必须位于 `remote.project_dir` 下。
- cleanup 前输出 dry-run 清单，并把清理结果写入报告。

### 6. 中危：checkpoint 非原子写入，损坏后会破坏恢复链路

位置：

- `gpt_small/training/train.py:65-80`
- `gpt_small/training/train.py:194-197`
- `gpt_small/training/sft.py:358-361`

证据：`save_checkpoint()` 直接 `torch.save(payload, path)` 写目标文件；训练循环会覆盖 `latest.pt`。

触发场景：保存时进程被 kill、断电、磁盘满、远程文件系统异常。

影响：`resume=True` 下一轮加载半写入 checkpoint 会失败；如果 `latest.pt` 被覆盖损坏，最后一个可恢复点可能丢失。

主持人裁决：保留为中优先级可靠性问题。

建议：

- 保存到同目录临时文件，例如 `latest.pt.tmp`。
- 写完并关闭后用 `Path.replace()` 原子替换目标。
- 可选保留 `latest.prev.pt` 作为兜底。

### 7. 中危：`torch.load` 和 tokenizer pickle 反序列化依赖可信文件

位置：

- `gpt_small/generate.py:74`
- `scripts/chat_checkpoint.py:81`
- `gpt_small/training/sft.py:67`
- `gpt_small/training/sft.py:207`
- `gpt_small/training/train.py:119`
- `gpt_small/tokenizer/bpe_tokenizer.py:44-47`
- `scripts/tokenize_corpus.py:91-94`

证据：多处直接 `torch.load(...)`；自定义 tokenizer 的 `vocab.bin`、`merges.bin` 使用 `pickle.load()`。

触发场景：加载来自第三方、云端同步目录、共享目录或被替换的 `.pt` / tokenizer 文件。

影响：在不可信文件进入加载路径时，可能通过反序列化执行恶意代码。

主持人裁决：保留为中优先级硬化项，并限定 threat model：只有 checkpoint/tokenizer 可由不可信来源提供时才是安全漏洞；纯本地自用时风险较低。

建议：

- 显式使用安全加载选项，并固定/记录 PyTorch 版本。
- 只接受可信来源 checkpoint；长期考虑 `safetensors`。
- tokenizer artifacts 改为 JSON、npz、MessagePack 等非 pickle 格式，或至少加 SHA256 校验。

### 8. 中危：远程进程监控只检查 PID，PID 复用会误判

位置：

- `scripts/sft_harness.py:196-209`
- `scripts/sft_harness.py:212-224`

证据：`remote_process_running()` 只读取 pid 文件并执行 `kill -0 "$pid"`；`kill_training()` 也只按 pid 操作。

触发场景：训练进程退出后 pid 被系统复用，或 pid 文件来自旧任务。

影响：harness 可能一直等待到超时，或误 kill 无关进程。

建议：

- pid 文件同时记录启动时间、run name、config path 和命令签名。
- 检查 `ps -p "$pid" -o args=` 是否匹配当前训练命令。
- 启动后短延迟检查 stderr 和进程状态。

### 9. 中危：过长 SFT 样本从头截断，可能截掉 assistant 训练目标

位置：`gpt_small/sft_data.py:201-208`

证据：超过 `context_length + 1` 时，代码保留 `input_ids[:max_tokens]` 和 `labels[:max_tokens]`。

触发场景：长 system/user 内容后跟短 assistant 答案，或多轮 messages 过长。

影响：真正需要训练的 assistant 标签可能被截掉，样本被静默跳过；也可能只训练答案前缀，削弱 stop/eot 和短答能力。

建议：

- SFT 数据优先保留包含 assistant 标签的尾部窗口。
- 或构建数据时拒绝超长样本并统计。
- 将截断数、跳过数、可训练 label token 数写入日志。

### 10. 中危：失败报告和 failure memory 中的中文建议文本已乱码

位置：

- `scripts/eval_sft_outputs.py:382-394`
- `scripts/eval_sft_outputs.py:453-475`
- `README.md`
- `项目进程.md`

证据：多个中文字符串显示为 mojibake；`advice_for_rule()` 会把乱码建议写入 failure memory。

触发场景：gate 失败后写 report 和 `reports/sft/failure_memory.jsonl`。

影响：历史失败经验不可读，下一轮复盘质量下降；项目文档也降低可维护性。

建议：

- 修复源文件编码和已损坏字符串。
- 增加小测试：断言 `advice_for_rule("identity")` 等输出包含可读中文关键词。
- 文档统一 UTF-8，并在编辑器/PowerShell 环境中固定编码。

### 11. 中危：远程 stdout/stderr 没有进入报告

位置：

- `scripts/sft_harness.py:227-250`
- `scripts/sft_harness.py:281-299`

证据：`download_artifacts()` 只下载 generation eval 和 sft log；`finish_report()` 不写远程 stdout/stderr tail。

触发场景：远程 import error、CUDA OOM、路径错误、checkpoint 加载失败。

影响：本地 report 只有 incomplete/failed 摘要，缺少第一现场，排障需要手动 SSH。

建议：

- 下载 stdout/stderr 的尾部 N 行。
- 在 markdown report 中写入 stderr tail、stdout tail、pid、退出状态线索。

### 12. 中低：SFT batch 动态长度会带来性能和显存波动

位置：

- `gpt_small/sft_data.py:221-241`
- `gpt_small/training/sft.py:220-221`

证据：每批按 sampled rows 的最大长度动态 pad；如果 `compile=True`，shape 频繁变化可能触发反复编译或吞吐下降。

触发场景：样本长度分布差异大，或开启 `torch.compile`。

影响：GPU kernel/compile cache 不稳定，显存峰值和吞吐难预测。

建议：

- 固定 pad 到 `context_length + 1`，或做长度 bucket。
- 若保留动态长度，默认禁用 SFT compile，并记录 batch shape 分布。

### 13. 中低：checkpoint eval 使用外部 config 构建模型

位置：

- `scripts/checkpoint_generation_eval.py:40-42`
- `gpt_small/training/sft.py:67-68`

证据：独立 eval CLI 用 `config["model"]` 构建模型，再加载 checkpoint 权重。

触发场景：eval config 与 checkpoint 内 `config.model` 不一致。

影响：可能加载失败，或非 shape 字段漂移导致评测环境与 checkpoint 元数据不一致。

建议：

- 优先使用 checkpoint payload 内的 `config.model`。
- 若外部 config 存在，显式校验两者一致。

### 14. 中低：下载与上传链路缺少内容哈希校验

位置：

- `scripts/download_hf_parquets.py:22-29`
- `scripts/download_hf_parquets.py:91-107`
- `scripts/upload_cache_to_autodl.py:77-99`

证据：下载脚本信任镜像 API 返回路径和内容；上传脚本跳过远端文件时主要按大小判断。

触发场景：镜像/API 被篡改、路径包含 `../`、远端同大小坏文件存在。

影响：训练数据投毒、缓存污染，极端情况下可能写出输出目录。

建议：

- 固定可信 dataset revision。
- 路径归一化后校验必须位于 output dir 内。
- manifest 记录 SHA256，上传/下载后验证。

### 15. 低危：chat 默认采样参数与 gate greedy 模式不一致

位置：`scripts/chat_checkpoint.py:33-35`

证据：chat 默认 `temperature=0.35`、`top_k=50`；正式 gate 多使用 `top_k=1` 的 greedy 模式。

触发场景：人工用 chat 脚本体验 checkpoint。

影响：人工观察结果和 harness gate 结论可能不一致。

建议：

- 增加 `--greedy` 开关。
- 或默认与当前 gate 一致，让采样模式显式 opt-in。

### 16. 低危：轮次脚本和配置重复较多，后续维护成本上升

位置：

- `scripts/run_v4176_refusal_core_repair.py:117`
- `scripts/run_v4177_preheldout_consolidate.py:202`
- `scripts/run_v4178_preheldout_mainline_gate.py:33`
- `experiments/v418/eval_v418_00_formal_heldout.yaml:29`

证据：多轮脚本重复 remote、monitor、cleanup、report、build_config、build_experiment；大型 eval YAML 内联大量规则。

触发场景：修改远端参数、清理策略、报告路径或通用 gate 规则。

影响：小改动需要跨多个历史脚本同步，容易漏改；diff 难看出本轮真正变化。

建议：

- 抽出公共 builder，例如 `scripts/sft_round_common.py`。
- 历史 `run_v*` 文件只声明本轮差异。
- 大型 rules 拆到独立 JSON/YAML，通过 `rules_path` 引用。

## 分角度小结

功能性：

- 评测产物完整性是最大风险，会直接影响 checkpoint 选择。
- SFT 超长样本截断策略偏粗糙，可能丢失训练目标。

可靠性：

- checkpoint 保存缺少原子性。
- PID 监控和远程日志下载不足，会降低长任务恢复和排障能力。
- SFT batch 动态 shape 对性能有潜在影响。

安全性：

- SSH 自动信任主机密钥不适合携带密码的远程训练链路。
- harness 的 YAML 执行权限过大，本地和远程都有命令执行面。
- `torch.load` 和 pickle tokenizer 需要明确可信来源或改用安全格式。

简洁性：

- 当前文档和可复现材料不同步。
- 轮次脚本重复多，大型 YAML 规则内联，后续维护成本偏高。

## 建议修复路线

第一阶段：保护 checkpoint 判断链路

- 清理/去重 generation eval。
- prompts 缺失 fail fast。
- generation eval 增加完成标记或原子产物协议。
- 报告写入唯一 prompt 覆盖数和重复数。

第二阶段：恢复可复现和流程合规

- 补齐 V4.18.9 及之后轮次的策略、YAML、报告和小型元数据。
- experiment 强制声明上一轮 report、failure memory、strategy memo。
- harness 启动前校验并摘要写入报告。
- 修复 failure memory 中文乱码。

第三阶段：收紧 harness 安全边界

- 去掉 `shell=True` build command。
- 禁止任意远程 `train.command` 或限制为白名单。
- 改用 SSH known_hosts/指纹校验。
- cleanup 加路径前缀检查和 dry-run。

第四阶段：提高恢复和维护质量

- checkpoint 原子保存。
- 远程 stdout/stderr tail 入报告。
- 抽出公共 round builder。
- 拆分大型规则文件。
- 统一 README 常用命令和当前推荐起点。

## 附：已知非问题

- 未发现硬编码真实密码或 API key；只发现 `AUTODL_PASSWORD` 等环境变量名和示例占位。
- `python -m compileall -f gpt_small scripts` 通过，说明当前 Python 文件没有语法级阻断。
- `scripts/chat_checkpoint.py` 是未跟踪文件，审查中纳入风险项，但本报告未修改它。
