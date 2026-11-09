# 强教师 trace → Qwen 27B SFT：首版数据交付方案

更新：2026-11-09，Asia/Shanghai。用途：针对 11 月 9–13 日目标工作周的交付方案与项目取舍；研究依据集中维护于 [trace 筛选与 SFT 专题](trace-sft-curation-research.md)。本页是方案，尚未读取本批原始 trace、实现转换、运行评估或训练。此前推荐卡片业务资料不等同于本批执行记录。

## 1. 本轮确认的目标与边界

用户持有 dsthinking 模型控制的类 OpenClaw harness trace，希望将强教师能力迁移到 Qwen 27B；11 月 6 日提出在下一工作周交给负责 SFT 的同事。用户进一步确认：

| 项目 | 当前认识 |
| --- | --- |
| 已有记录 | 包含模型实际输入、thinking、工具调用与返回；具体原始格式尚未读取 |
| 同题候选 | 每个 case 只有一条 trace；用户任务与背景各不相同，不能视作同题多个候选 |
| 任务质量 | 任务偏生成式，最终效果评估尚未运行；不能默认存在可靠成败标签 |
| 运行环境 | harness 能执行，但环境尚未搭好 |
| 数据规模 | 尚待确认，按 demo 阶段设计 |
| 本轮优先级 | 先跑通数据到 SFT 的流程，质量优化以后推进 |
| 训练接口 | Qwen 27B 的具体版本、训练框架、上下文长度和 thinking 策略待用户与同事确认；现按最简方案推进 |

`dsthinking` 按用户给出的教师标识记录，不擅自等同某个 DeepSeek checkpoint。Qwen 27B 也不自行指定为 3.5、3.6 或 3.8。保留 11 月 9–13 日工作周的候选节奏，具体交付日尚未确定；不因讨论日期推进而自动顺延一周。

**首版建议：离线序列蒸馏数据转换，加基础可训练性检查。** 交付目标是同事能加载并对预期 assistant 输出计算 SFT loss。语义质量状态为 `not_evaluated`；不等待结果评审器、学生模型采样或 harness 重放环境上线。

教师强是用户对来源的描述，不能据此把每条记录标为成功。质量尚未评估不影响这次 demo 数据交付，但报告要保留这一事实；跑通也不直接证明能力迁移成功。

## 2. 补检索后的技术选择

此次按机制补查离线教师模仿、学生适配选择、示范再生成、学生错误纠正、过程监督和在线分布蒸馏，避免仅搜索 Agent 轨迹评分。

| 方向与依据 | 当前能借鉴什么 | 本次取舍 |
| --- | --- | --- |
| 离线序列蒸馏：[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[Agent-FLAN](https://github.com/InternLM/Agent-FLAN) | 前者提供强教师生成数据微调小模型的实例；后者强调 Agent 数据格式与能力学习的关系。已有文本／动作即可做 SFT，不需要教师 logits | 作为首版主路线；不将其 benchmark 收益当作本项目保证，也不套用旧模型模板 |
| 学生适配选择：[SmartAD](https://aclanthology.org/2026.findings-acl.1349/) | 同题多个正确教师候选，用学生逐 assistant 回合的平均 token NLL 排序，再对回合平均；另有分段加权训练 | 当前已确认每题只有一条，选择模块无候选可比较，撤下“下一阶段优先”建议。分段加权可单独研究，需训练方配合。论文主要验证 1.5B／3B 的检索与数学任务，未直接验证本场景；未确认官方代码 |
| 学生适配再生成：[Finetuning with Sampling](https://github.com/aakaran/finetuning-with-sampling) | 可以从每题一条专家示范出发，利用学生生成与概率调整候选，保持普通 SFT | 原论文是离线文本采样，不依赖 Agent harness。迁移时若改工具动作，需执行并取得新返回；固定观察下的末轮回复适配可离线研究。暂不加入 demo 主链，理论与近似代码区别见研究专题 |
| 学生错误纠正：[SCoRe 论文](https://proceedings.mlr.press/v306/lyu26j.html)、[代码](https://github.com/haruhi-sudo/SCoRe) | 学生先尝试，教师纠正最早关键错误，得到面向学生弱点的数据；公开实现含冷启动、SFT 和后续 RL | 等学生和环境可运行后借鉴其纠正数据生成模块；论文整体收益含 RL，不能全归给 SFT |
| 过程诊断与监督：[TQS](https://github.com/DuHeng0128/TQS)、SRFT／SWE-Prime | 可计算过程特征；对可靠步骤监督 | 后续作为评分或监督策略接入。第一版只处理结构问题，报错与重复只记录，不据此直接判坏或普遍屏蔽 |
| 在线分布蒸馏：[TRL DistillationTrainer](https://huggingface.co/docs/trl/main/distillation_trainer) | 对学生在线生成使用教师分布反馈 | 需要额外教师评分、训练与词表／tokenizer 适配；现有文本 trace 本身不足以完成该算法，暂不列入数据 demo |

2026-11-09 修正：此前将 SmartAD 列为优先的轻量扩展，依赖尚未确认的同题候选条件。用户现已明确每题只有一条，因此当前主线保持单轨迹离线蒸馏，后续先研究质量与覆盖。学生 NLL 可以作为可选诊断，不要求 v0 计算，也不据此在不同任务间直接取最低 loss。

### 2.1 单 case 单 trace：两种方法分别需要什么

| 用法 | 一条原始 trace／case 是否能起步 | 是否需要工具执行环境 |
| --- | --- | --- |
| 直接使用教师 trace 做 SFT | 可以，无须同题重复 | 离线转换与训练数据制作不需要 |
| FWS 原论文的文本示范再生成 | 可以；算法在处理时生成新候选，不要求日志已有多个候选 | 不需要 harness；需要参考学生的生成、概率计算及信息保留检查 |
| FWS 启发的固定观察、末轮回复适配 | 可以；只替换在相同真实输入下的最后一轮回复 | 可离线；仍要核验事实、约束及与记录的一致性，这是迁移变体 |
| FWS 启发的工具动作／路径再生成 | 一条可信示范可作为起点 | 需要恢复相应状态并执行改动后的动作及受影响后缀 |
| SmartAD 的同题轨迹选择 | 单候选时退化为保留原候选，没有选择作用 | 已有完整多候选时可离线评分；重新采集交互候选通常需要环境 |
| SmartAD 的分段加权监督 | 可以单独借鉴，不依赖多个候选 | 本身不需要环境；需要训练器支持，不能视为完整 SmartAD 的复现 |

SmartAD 论文 §3.2 实验每题采样 `K=10`，核心条件是存在多个可接受候选，10 不是本项目必须遵守的数量。这里“同题”应固定任务、用户背景／记忆、可用工具及初始环境；相似任务或不同用户记录不能冒充同一 case 的候选。将来可以在冻结案例上补采，不要求真实用户重复发任务；但模型请求记录本身不等于可恢复的文件、账号或网页状态，是否能补采需另行核验。

FWS 的文本采样与 harness 改造是两件事。例如原记录调用 `read_file("A.csv")`：基于真实返回重写最后的总结，不必重新读文件；若换成读取 `B.csv`、改变筛选代码或写入动作，必须取得对应执行结果，不能接上 A 的旧返回。已有 harness 若支持状态恢复、分支续跑和完整记录，可直接复用；缺什么补什么，不预设整体重建。不可变只读工具的缓存只有在输入与状态等价可核验时才可复用。

固定工具动作并不自动保证任意 thinking 改写都有效：改写可能泄漏未来观察或改变后续输入。离线探索优先限制在固定完整上下文的末轮回复，单独标记来源；其证据仅覆盖回复适配，不代表学会新的工具策略。当前未评估的教师记录也不能自动当作 FWS 所需的可信专家答案。

## 3. 哪些开源项目以什么形式使用

| 项目 | 使用形式 | 首版成本与扩展性 |
| --- | --- | --- |
| [Agent Data Protocol](https://github.com/neulab/agent-data-protocol) | 参考 `schema/atif.py` 的 reasoning、tool call、observation、来源和关联设计；本 harness 只写一个输入适配器 | 低到中。首版可保留原请求结构，用轻量记录包装；ATIF 互转以后接入。工具名／参数不因套通用转换器而自动改写 |
| [Transformers](https://huggingface.co/docs/transformers/chat_templating) | 在具体 checkpoint 确认后，用其 tokenizer／processor 与 chat template 做编码、长度及格式检查 | 低到中。以目标配置控制版本和模板，换学生时替换导出配置 |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 或 [ms-swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Agent-support.md) | 只对接同事正在使用的一个训练器的数据读取与模板入口，不在数据工程内重新搭训练器 | 确认后实现一个 exporter。TRL 可用 conversational prompt-completion；ms-swift 可按其数据／模板入口转换，并检验实际监督位置 |
| TQS | 首版只参考特征定义，后续可选评分函数 | 不移植其全部编码任务权重；无编辑文件、短轨迹和合法重试不自动淘汰 |
| SmartAD／Finetuning with Sampling／SCoRe | 分别预留选择器、再生成器、学生纠错采集器 | 不作为首版依赖，不影响转换程序交付；各自增加所需模型或环境接口 |

建议最少依赖为 Python 标准库读写 JSONL，加 Pydantic 校验；目标模型确认后使用 Transformers。Parquet、Hugging Face Datasets、大规模调度或数据库按实际数据量增加。参考代码如复制进项目，需要保留相应许可和来源；不为 demo 搬入整套仓库。

## 4. 最小流程与样本单位

```mermaid
flowchart LR
    A[保留原始 trace] --> B[提取真实模型请求与输出]
    B --> C[结构检查及按任务分组]
    C --> D[目标模型格式导出]
    D --> E[同事侧编码和训练读取验收]
    C -.后续.-> F[质量评价与学生适配]
    F -.重新导出.-> D
```

**原始保存与分组单位为任务／session；优先的训练单位为一次真实模型请求及当次 assistant 输出。** 已有实际请求时，这种方式能自然保留工具反馈、上下文压缩、skill 动态注入和多角色差异，不需要从整段日志猜历史。模型当次可见的 system、user、历史 assistant、工具返回均属于 prompt。

对于连续调用：第 1 条样本可以学习 thinking 和读文件调用；第 2 条样本的 prompt 包含真实读文件返回，学习下一次思考和动作；最后一条样本学习最终交付。每条只监督当次输出，不能对重复出现的历史 assistant 再次计算损失。对同一任务的采样权重先如实报告，之后可控制长任务占比。

这只是把一条轨迹展开成多个决策位置，**不产生同一 case 的多条替代轨迹**，也不满足 SmartAD 的同题候选选择条件。每题只有一条不妨碍普通 SFT 学习这些实际决策。

逐请求方式会重复输入前缀，增加训练成本。demo 优先保证对齐；将来如确认日志是简单追加、上下文一致，可增加完整 conversation 导出，以一次编码监督多个 assistant 回合。两种视图共享原始来源与任务划分。

建议一个标准记录仅包含：

```text
sample_id、trace_id、request_id、source_ref
prompt：当次真实输入消息
completion：当次模型输出，分开保留 thinking、content、tool_calls
tools：当时的工具定义；如嵌在输入中，保留原表示和来源
metadata：教师／harness 版本、缺失或截断状态、quality_status
```

`quality_status=not_evaluated`；不补造成败。标准层保留原字段，并不要求下游训练器直接认识所有字段。训练导出列与审计元数据可以按 `sample_id` 分开，避免把评价内容混进 prompt。

### 4.1 一个虚构的导出草案

下面示意 conversational prompt-completion，字段与 thinking 渲染需由目标 exporter 核验，尚不是已经验收的 Qwen 模板。目标是学习发起调用；这次调用的结果应出现在下一次真实请求的 prompt 中。

```json
{
  "prompt": [
    {"role": "user", "content": "查看工作目录里有哪些费用表，先列出文件。"}
  ],
  "completion": [{
    "role": "assistant",
    "reasoning_content": "先列出目录，确定实际文件名。",
    "content": "",
    "tool_calls": [{
      "id": "call_demo_1",
      "type": "function",
      "function": {"name": "list_files", "arguments": {"path": "."}}
    }]
  }],
  "tools": [{
    "type": "function",
    "function": {
      "name": "list_files",
      "description": "列出指定目录的文件名。",
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  }]
}
```

这是虚构字段演示，不由本助手伪造工具返回，也不表示执行过或质量已评分。

### 4.2 首版保留什么、检查什么

1. **保留记录到的 thinking、动作、工具反馈和最终回复。** reasoning 独立保存在标准层，再由 exporter 映射到学生约定的表示。初版不批量改写、压缩思考或删除恢复步骤；用户同事若确定只训非 thinking 模式，改导出策略并明确能力目标。
2. **检查样本能否正确参与训练。** 输入／输出可解析、目标非空、工具参数结构有效、调用返回可关联、角色不混淆。assistant 的 content 为空但有 tool_calls 是合法目标，不按空字符串删除。
3. **只隔离无法忠实转换的记录。** 不完整流式输出、引用缺失且无法解释、工具定义缺失影响学习等，写明原因；不补造。工具 error 返回本身不等于结构损坏；能够忠实表示的错误和恢复保留。
4. **保留未评估状态。** 不要求任务成功，不按 LLM 总分、长度或学生困惑度淘汰。可对完全重复的同一请求记录合并出处；不删除不同上下文下恰好相同的动作。
5. **按任务／会话及可识别的派生家族切分后，再展开逐请求样本。** 同一任务的前后回合放同一侧。样本太少时保留单独的格式测试样例，并说明还没有有效独立开发集，不能拆回合伪造泛化验证。
6. **按学生真实 tokenizer 检查长度。** 教师 token 数不能代替学生 token 数；超长请求单独记录并等待分窗策略。不得随机截掉必要上下文或把半个工具调用训练为完整输出。具体长度未知时先保留原记录和字符统计。

如果存在图像／其他附件或子 Agent，保留引用和角色，按目标训练器实际支持范围分流；不能悄悄把多模态任务当成完整文本任务。

## 5. 让程序容易修改的最小组织

以下是建议目录，不表示已实现：

```text
trace_data/
  schema.py        # 记录定义与基础校验
  normalize.py     # 本 harness → 标准记录
  select.py        # 结构分流、去重、任务切分；之后可接质量规则
  export.py        # 标准记录 → 同事所需格式；可选 tokenize 验收
  config.json      # 字段映射、目标配置、切分种子、启用规则
```

原始数据、标准记录和训练导出分别保存，不覆盖。评审结果后续通过 `sample_id / trace_id` 关联；更换筛选规则只重做选择和导出，更换 Qwen checkpoint 只更换目标配置及编码验收，增加学生采样则写入新来源记录。先用函数参数和配置实现这些边界，无需插件注册系统、服务端或工作流平台。

首版形成一个命令入口即可；执行日志记录转换版本、输入校验值、配置和各分流计数。真实数据沿用项目隐私约束，不保存到公开 Git 仓库。

## 6. 给 SFT 同事的最小交付物与验收

| 交付物 | 内容 |
| --- | --- |
| `train.jsonl`、条件允许时的 `dev.jsonl` | 同事训练器能读取的目标格式；按任务隔离 |
| `manifest.json` | 数据和转换版本、目标 checkpoint／模板、切分、样本与 token 统计、质量未评估说明 |
| `sample_index.jsonl` | 训练行到 trace／request／原始记录的映射；可承接后续质量标注 |
| `report.md` | 字段覆盖、缺失、超长、重复和隔离原因，以及少量结构检查例子 |
| 导出配置与最小加载例子 | 明确 prompt／completion 边界、thinking 和工具格式，以及实际 loss 范围 |

接口未确定前，可先产出中立记录和预览；要称为“同事可直接使用的 SFT 数据”，交付前仍需使用其指定模板做一次编码验收。用户已表示后续确认模型和框架，不重复要求现在定案。

验收只需覆盖影响使用的关键项：

1. 数据读取与 batch 构建成功，所有导出行经过结构检查；隔离数和原因可回查。
2. 解码实际 `input_ids / labels` 抽查：当次 thinking（如约定训练）、工具名与参数、最终回复属于目标；prompt 中的历史消息、system／user、工具返回不被当作本次预测目标。
3. 模板没有丢失 thinking 或 tool_calls，没有添加错误停止位置；训练和部署的工具解析约定一致。
4. 同事做一次小 batch 的前向 loss／短程训练检查，确认目标 token 非零、loss 有限、可正常更新。该检查由训练方运行，本轮未执行。

**训练器配置不能只看名字。** 本轮读取的 ms-swift 文档中，`all` 会训练所有 token，`last_round` 指最后一个 user 后的内容，不能直接等同“最后一次模型调用”。TRL 的 prompt-completion 可仅监督 completion，但最终仍需验证该 checkpoint 的 thinking／工具模板和实际 label mask。不要把某个开关自动当成通用正确配置。[TRL 文档](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md)、[ms-swift 参数](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)

开发集的 teacher-forcing loss 可验证数据与训练流程，不代表 Agent 执行效果；实际生成任务评价等后续环境和标准接入后验证。

## 7. 11 月 9–13 日的候选节奏

| 时间 | 工作 | 应得到的具体结果 |
| --- | --- | --- |
| 周一 11/9 | 看少量覆盖 thinking、工具、最终回复的原始请求，确认字段映射；与同事确定实际导出目标 | 一个完整转换样例和数据接口说明 |
| 周二 11/10 | 实现标准化、结构分流、分组切分和单一 exporter | 一小批可读取的数据及转换统计 |
| 周三 11/11 | 同事进行编码／label 抽查和小 batch 试读，修正格式问题 | 经训练入口验证的 demo 包 |
| 周四–五 11/12–13 | 扩展到本次确认的数据范围，整理索引、配置和报告 | 可追溯的 v0 数据包；是否扩大量由真实数据规模决定 |

时间是建议安排，不是未经数据检查的完成承诺。若字段简单可更早交 demo。原始数据到位与训练目标确认影响对应环节；语义评分、学生采样和 harness 搭建不作为 v0 依赖。

当前扩展主线是：**单 trace 离线数据接通 → 生成式任务质量检查与覆盖分析 → 根据学生表现定位数据缺口。** 质量检查先对结果是否满足请求、事实是否有输入依据、工具状态与最终描述是否一致等做小批校准；不要求先建立庞大总分系统。

候选分支按问题启用：需要改善固定观察下的交付时，可研究 FWS 启发的末轮适配；拥有同题多条可接受轨迹时，再比较 SmartAD 选择；需要改变工具策略且具备环境时，再研究重新执行、学生纠错或动作再生成。这些分支不是必须依次完成的版本阶梯，不为使用论文而先制造多候选或重建 harness。

## 8. 新增资料核验记录

均为 2026-11-06 公开来源阅读，未安装或运行项目。原论文和代码链接已在上文对应主张处提供；本表的 SHA 是文件对象，非锁定整个仓库版本。

2026-11-09 针对单 case 单 trace 条件，复核 FWS 正文 §3–4 和 SmartAD §3.2–3.3；本次仅更新适用条件与方案优先级，未生成新候选或实测迁移收益。

| 来源 | 本轮核验程度 |
| --- | --- |
| SmartAD，Findings ACL 2026 | 阅读论文方法与选择公式：同题正确候选、逐回合 NLL、分段权重；未确认官方代码，不声称是可直接安装的工具 |
| SCoRe，ICML 2026 | 会议原始条目；经 Software 链接定位官方仓库并读取 README，SHA `c16f5ecc94ac19155b86d9cade0091c5ef0b6c34`；未复现其 SFT／RL |
| ADP | `schema/atif.py`，SHA `6b68ac5ee9d54cb4f39f4963f19d6025c7009bae`；包含 reasoning_content、调用返回关联及 extra，不代表能保存所有 harness 请求细节 |
| TRL | `docs/source/sft_trainer.md`，SHA `b34d78602f9e12f0b7c042d0a2daec319de3ba20`；`distillation_trainer.md` 前段，SHA `4ba1255b61236f91346afdc6772bf7c9e71a19cc` |
| ms-swift | Agent-support 文档 SHA `6cf6b5ce18a739b8fe0d00d839480384e8e69b4d`；参数文档 SHA `0538a0e758cb726dad78f2e692f8db39ee8721bf`，重点核验 loss 与 thinking 设置 |
| Agent-FLAN | README，SHA `e2d7b75bfb15aabeb66d6abe2f6278d8a49dec5b`；用作方法参考，不将其 Llama2 格式复制给 Qwen |
| DeepSeek-R1 | README，SHA `40d1af36fdb96e26129527b0531bb3ac195f6713`；仅作为离线蒸馏实例，不用于认定本批教师身份 |
| Qwen | 核验 [Qwen3.5-27B 官方卡片](https://huggingface.co/Qwen/Qwen3.5-27B) 的 thinking 与工具接口说明；模型系列存在不同版本，不能从“27B”唯一确定配置 |
