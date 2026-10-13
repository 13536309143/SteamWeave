# SteamWeave 技术手册

[English](TECHNICAL_MANUAL.md) · [项目概览](../README.zh-CN.md) · [实验协议](EXPERIMENT_PROTOCOL.md)

## 1. 任务与信息边界

SteamWeave 根据观察时刻 (t) 已经获得的信息，预测设备在下一次计划巡检时的退化水平。目标采用数据包的 `nextinslevel.txt`，单位为任意退化单位 adu；时间采用任意时间单位 atu。本项目的预测量不等同于日历失效日期或剩余寿命。

$$
\hat y_{i,t}=f(x_{i,t}^{num},x_{i,t}^{img},x_{i,t}^{text},H_{i,\le t},T_{next}-t).
$$

`H` 仅包含同一设备先前的输入观察。目标标签，包括过去记录的目标标签，不进入特征重建或 LLM 请求。系统保留数据集的目标约定，不补造源文件未明确说明的巡检与维修先后关系。

## 2. 项目结构与运行环境

| 目录 | 职责 |
| --- | --- |
| `sg_agents/` | 数据协议、时间重建、专家、融合、供应商适配与 API |
| `static/` | 本地界面、历史图、上传和解释 API 选择 |
| `research/` | 深度结构重建、固定实验协议、压力测试与公开特征资产 |
| `reports/` | 配置、结果、配对分析与独立图表 |
| `config/` | 不含凭证的供应商环境示例 |
| `tests/` | 信息边界、缺失数据、协议与网络结构测试 |
| `Multimodal data for prognostics/` | 原始数据集，保留源文件 |
| `artifacts/` | 生成的模型、缓存、日志与进程文件，不提交 Git |

已测试环境为 Python 3.14、macOS ARM64、scikit-learn 直方图提升模型，以及可选的 PyTorch MPS。普通 Python 服务也面向 Linux；Windows 可使用 WSL 或 Docker。当前未实现原生 Windows 文件锁。Docker 为部署配置，当前电脑没有可用于实际验证的 Docker 运行环境。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m sg_agents.cli train --offline
.venv/bin/python scripts/service.py start
```

仓库附带的语义资产可让默认 MiniMax 特征流程离线训练。新报告提取和实时解释需要相应 Key。界面地址为 `http://127.0.0.1:8787`，API 文档位于 `/docs`。macOS 可双击 `scripts/start.command`。后台服务可以在关闭终端后继续运行；重启电脑后，除非安装可选的登录启动配置，否则需要重新启动。

## 3. 数据协议与审计

每个设备目录包含时间、文本、目标和四个数值向量。第 `X` 条观察对应 `Image/X.tiff`，图片从 1 开始编号。各字段长度必须一致；观察时间必须有限且严格递增；次数和持续时间必须有限且非负。

本地副本有 50 个目录、7,626 条记录。文本去除外层引号后，794 个原始字符串归并为 766 个规范化唯一字符串。设备 1–20 与 31–50 的时间、数值和文本轨迹分别相同，因此按 30 个输入轨迹组评估。分组不使用标签，也不把 TIFF 元数据差异视为独立模拟证据。这是对本地副本的审计，不代表原论文采用了同一个数据版本。

轨迹分组仅用于离线划分，组 ID 不作为预测特征。原始记录全部保留；泄漏控制训练中，同组目标巡检事件按记录数反向加权，降低重复目录和高频观察的重复计权。

## 4. 时间重建与图片处理

根据当前数值字段反推下一巡检与最近维修时点：

$$
T_{next}=t+\Delta t_{next},\qquad T_{repair}=t-\Delta t_{repair}.
$$

最近维修时点向前更新时，识别一次已观测维修变化；清洗次数增加、机械清洗计数复位提供补充信息。两次观察之间的全部事件不一定能恢复，因此保留事件时间区间，不编造精确时间。

文本保存首次出现时间、当前连续重复段起点、两种观测年龄、重复标记，以及重复段开始后是否发生维修。真实报告生成时间仍为未知。相同措辞重新出现时，不自动沿用上一次的生成时间。

曲线编码器定位绘图区，排除坐标轴与刻度，再检测曲线。只有坐标轴的 TIFF 即使存在，也属于缺失证据。可用曲线输出 32 个采样特征和 8 个形态统计量。高度使用统一绘图区归一化，保留不同记录之间的幅值差异；这些是图片中的近似几何特征，不是恢复的原始信号。不可读图片或无法定位坐标框时，明确标为不可用。

## 5. 七类有明确协议的 Agent 职责

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| 调度 | 当前请求、同设备历史 | 当前时点上下文、模态可用性 |
| 数值专家 | 四个原始数值字段 | 监督学习点估计 |
| 曲线专家 | 曲线几何、预测跨度、距维修时间 | 点估计或不可用状态 |
| 文本专家 | 六个语义字段、64 维词汇哈希、跨度与距维修时间 | 点估计与证据来源 |
| 维护专家 | 数值与 13 个重建状态字段 | 点估计 |
| 融合 | 专家预测、数值、12 个质量/上下文字段 | 点预测与区间 |
| 审核解释 | 白名单内的预测和证据 | 程序质量标记、可选 LLM 解释 |

语义字段区分维修前严重程度、维修后严重程度、机械清洗、化学清洗、明确紧急性和维护类型未知。严重程度是语言中的序数特征，不转换成 adu。结构、值域和原文子串校验验证格式与证据来源，不保证每个语义标签都正确。仅大小写不同的引文会映射回原文中的准确子串。

流程采用明确协议与固定协作图。预测专家为监督学习回归器，语言模型负责提取和解释。七类职责不意味着每条记录需要七次付费 LLM 调用。

## 6. 专家与融合训练

直方图提升模型采用 180 次迭代、15 个叶节点、叶节点最少 25 个样本、学习率 0.065、L2 正则化 4。不使用随机拆分相关观察的内部早停。每个拟合分区独立移除完全没有观测值的列，预测沿用该分区的列选择；其余 NaN 保留给模型缺失值路径。列选择不访问验证或测试记录。

融合器使用四折内部验证生成的专家折外估计。外层与内层均按重复轨迹组划分。当前模态不可用时，相应专家预测为 NaN，并同时提供质量标记；空白曲线不会被当作正常曲线。

融合增强中，完整输入权重为 0.7，屏蔽当前曲线、当前文本、同时屏蔽两者分别为 0.1。缺失测试移除当前专家证据，已获得的历史维护信息仍可存在。融合器不是注意力模型，专家预测也不等同于因果贡献。

运行界面使用种子 20261013 的 21/3/6 轨迹组拟合、校准和测试划分。研究交叉验证与时间实验写入独立目录，不覆盖界面的预测模型。

## 7. 预测区间

原版预测头用独立的 5% 和 95% 分位数模型处理融合表示。在留出校准组中，对每个目标巡检事件取最大区间越界误差，再使用对应分位数扩展上下界。若上下分位数交叉，先按大小排序。

报告逐记录覆盖率、逐事件覆盖率及区间宽度。这里的标称 90% 属于经验校准：同一轨迹内事件相关，未来时段分布也可能改变。运行协议没有自动假定物理截断范围，因此下界可能为负，不进行无提示裁剪。

## 8. API 供应商、凭证与缓存来源

| 供应商 | Key 环境变量 | 模型变量 | 地址变量 |
| --- | --- | --- | --- |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_MODEL` | `MINIMAX_BASE_URL` |
| OpenAI GPT | `OPENAI_API_KEY` | `OPENAI_MODEL` | `OPENAI_BASE_URL` |
| 千问 | `DASHSCOPE_API_KEY` | `QWEN_MODEL` | `QWEN_BASE_URL` |
| 自定义兼容服务 | `CUSTOM_API_KEY` | `CUSTOM_MODEL` | `CUSTOM_BASE_URL` |

`LLM_PROVIDER` 选择提取供应商；`LLM_AUDIT_PROVIDER` 或解释请求中的 `provider` 独立选择解释供应商。例如保留 MiniMax 模型输入，单独使用 GPT 审核：

```dotenv
LLM_PROVIDER=minimax
LLM_AUDIT_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_key
OPENAI_MODEL=your-gpt-model
```

适配器不会回退使用其他供应商的 Key。Key 只在后端读取，不返回 JSON、不提交 Git。模型名称明确配置，不自动替换。千问地址和 Key 应属于同一地域与计费方案。

`*_API_STYLE=chat` 使用 Chat Completions；`responses` 提取 Responses 的最终 `output_text`。OpenAI Chat 默认使用 `max_completion_tokens`，其他预设默认 `max_tokens`；必要时用 `*_TOKEN_PARAMETER` 改动。`*_EXTRA_BODY` 可以添加模型参数，但禁止覆盖凭证、模型、输入和输出上限。Chat 的推理参数为 `reasoning_effort`；Responses 使用 `reasoning` 对象。只传递当前模型支持的参数。

```bash
.venv/bin/python -m sg_agents.cli check-api --provider minimax
.venv/bin/python -m sg_agents.cli check-api --provider openai
.venv/bin/python -m sg_agents.cli train --provider qwen
```

缓存按供应商、模型、地址和协议隔离。原始 M2.7 标准与高速别名使用已注明的兼容命名空间。切换提取命名空间需要重建特征并重新训练；只切换解释 API 不需重训。仓库默认特征缓存不会作为 GPT 或千问缓存使用。

MiniMax 已完成真实请求验证。GPT、千问与自定义接口完成传输协议和错误路径测试；没有各自 Key 时，不声称已验证真实账户可用性。官方参考：[OpenAI Chat](https://developers.openai.com/api/reference/resources/chat)、[阿里云地址概览](https://help.aliyun.com/zh/model-studio/base-url)、[MiniMax 模型调用](https://platform.minimax.cn/docs/guides/text-generation)。

## 9. 服务 API

| 接口 | 功能 |
| --- | --- |
| `GET /api/health`、`/api/providers` | 就绪状态与不含秘密的供应商配置 |
| `GET /api/summary`、`/api/devices` | 数据审计与评估概览 |
| `GET /api/devices/{id}/records` | 选择观察记录 |
| `GET /api/devices/{id}/history?through=t` | 仅展示不晚于 `t` 的观察预测 |
| `POST /api/predict` | 已有记录预测及当前模态屏蔽 |
| `POST /api/explain` | 白名单证据解释，可选供应商 |
| `POST /api/predict/custom` | 数值、文本、图片 multipart 上传 |
| `GET /api/export/{id}` | UTF-8 CSV 导出 |
| `POST /api/train`、`GET /api/train/status`、`POST /api/reload` | 训练生命周期 |
| `POST /api/provider/check?provider=name` | 实时接口验证 |

新观察仅使用严格早于它的同设备历史，新设备历史为空。图片最大 10 MB、2,000 万像素；数值必须有限、非负，清洗次数必须为整数。解释层不会接收展示层附加的评估目标。

已验证服务绑定 localhost，跨域写操作会被拒绝。缓存使用文件锁与原子替换，准备/训练任务使用非阻塞锁。公开 HTTP 部署需要自行配置身份和请求额度控制。

## 10. 实验命令与比较范围

```bash
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m research.glove
.venv/bin/python -m research.run_suite
.venv/bin/python -m research.extend_stress
.venv/bin/python -m research.summarize
```

实验包含三种子的五折轨迹组评估、三次未来时段评估、五折设备 ID 诊断。六项消融与配对压力测试覆盖种子 17 的全部五折。重建深度基线使用一个种子，不是三种子研究。CPU 可用 `--device cpu`；改变训练计划时使用新的 `--run-name`，避免沿用不同设置下的完成结果。

原论文的三分支 MAE/MSE/MRE 作为附有协议说明的文献参考。当前系统和本地结构重建使用相同分区重跑。数据实现、预处理、优化和硬件差异，使这些结果不能被称为原论文数值的精确复现，也不能直接用于跨论文优越性结论。

实际指标、置信区间、消融、压力测试、逐样本预测与训练预算限制见[实验报告](../reports/EXPERIMENTS.zh-CN.md)。

## 11. 验证与发布

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_deployment.py --with-minimax
.venv/bin/python scripts/check_publication.py --tracked
```

测试覆盖因果重建、标签隔离、空白图片、重复轨迹、整列缺失、供应商隔离与网络分支尺寸。实时服务验收覆盖预测、模态屏蔽、历史截止、上传、CSV 与真实解释。

发布原始数据、引用说明、代码、固定特征资产和实验输出；排除 `.env`、本地日志/PID、虚拟环境、运行缓存与模型二进制。附带语义资产仅含模拟报告证据，不含认证信息。

## 12. 限制与后续研究

这是含重复输入轨迹的模拟基准，轨迹组数为三十。LLM 语义标签不是人工验证真值。本副本中部分维修/文本年龄交互没有自然样本；合成延迟测试只能验证敏感性。深度重建有已披露的实现缺项与预算上限。任何模块的收益均需由消融验证。

后续应预先规定开发集上的模型选择，增加独立模拟种子或工业数据，审查语义标签质量，并同时评估推理成本与预测误差。外层测试组保留用于最终评估。

## 13. 可选的 v2 优化

v2 预测头用二次拟合描述当前图像曲线，并保留严格早于当前观察的同设备曲线描述、时间间隔和变化率。检测到维修后，在处理当前观察之前重置曲线历史。历史目标标签不进入特征。屏蔽当前模态仍保留过去已经获得的历史，这不同于整个历史流中断。

八个固定候选比较曲线几何/历史/语义特征、绝对误差或平方误差提升模型、Extra Trees 与多项式岭回归。三折内层轨迹分组验证按 MAE 选择候选；时间实验在训练期内采用三个前向巡检事件窗口，并剔除目标巡检跨越训练截止点的记录。外层校准组仅确定对称的事件最大误差区间；外层测试记录不参与候选选择。这是对已查看基准的探索性改善，不是新盲测数据集。可用性切换是在首次缺失曲线诊断之后加入的。时间候选没有改善当前副本，除非另有验证支持，建议保留原版时间检查点。

```bash
# 给已有且匹配的原版检查点添加优化头。
.venv/bin/python -m sg_agents.cli optimize --offline
# 从新克隆仓库训练两个层次。
.venv/bin/python -m sg_agents.cli train --offline --method optimized
# 采用前向内层选择的未来时段检查点。
.venv/bin/python -m sg_agents.cli train --offline --split temporal --method optimized
# 恢复原版预测方法。
.venv/bin/python -m sg_agents.cli train --offline --method legacy
```

原版检查点保留在已忽略的 `artifacts/baselines/legacy-v1/`。当前曲线可用时，v2 使用训练内选出的曲线/历史预测头，原版专家输出作为诊断；当前曲线缺失时，使用原版模态 dropout 专家融合。此可用性切换保留原版缺失曲线点预测。响应提供 `prediction_method`、`forecast_head` 与 `legacy_prediction_adu`，界面和审计警告明确说明区别。通过服务重新训练会保留当前方法；`POST /api/train?method=legacy|optimized` 可明确指定。

运行 `python -m research.optimize run` 复现嵌套比较，再运行 `python -m research.optimize summarize` 生成汇总。前者使用核心依赖，图表/报告生成还需要 matplotlib。[v2 报告](../reports/optimization/OPTIMIZATION.zh-CN.md)提供配对结果、逐折选择、原始预测与局限。公开的内层行索引按数据集观察顺序，用无损的左闭右开区间（`range(start, stop)`）编码。

方法参考：[直方图提升损失函数](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)、[分组交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)、[嵌套选择与评估](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。
