<p align="center"><img src="docs/assets/banner.svg" alt="SteamWeave — 多模态证据、协作 Agent 与未来退化预测" width="100%"></p>
<p align="center"><b>简体中文</b> · <a href="README.md">English</a> · <a href="docs/TECHNICAL_MANUAL.zh-CN.md">技术手册</a> · <a href="reports/EXPERIMENTS.zh-CN.md">实验报告</a></p>
<p align="center"><a href="https://github.com/13536309143/SteamWeave/actions/workflows/tests.yml"><img src="https://github.com/13536309143/SteamWeave/actions/workflows/tests.yml/badge.svg" alt="测试"></a> <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14"> <img src="https://img.shields.io/badge/Records-7%2C626-087f83" alt="7626 条记录"> <img src="https://img.shields.io/badge/Modalities-Numeric%20%2B%20Image%20%2B%20Text-23424d" alt="三类模态"></p>

**SteamWeave（蒸汽织维）** 是面向蒸汽发生器下一次计划巡检退化预测的维护感知多 Agent 研究平台。名称由 *steam* 与 *weave* 组成，表达将数值状态、信号图像、巡检语言和维护历史编织成可追溯预测流程的理念。

仓库包含原始模拟数据集、本地 Web 应用、模块化预测 Agent、可切换的 LLM 接口，以及已完成的实验、逐条预测和划分清单。时间与退化分别使用任意单位 **atu** 与 **adu**。

## 研究特点

| 能力 | 当前实现 |
| --- | --- |
| 专家协作 | 数值、曲线、文本与维护回归专家，利用折外预测训练融合模型 |
| 信息边界 | 仅使用观察时点已获得的同设备历史；标签不进入特征构造或 LLM 输入 |
| 缺失与陈旧信息 | 仅有坐标轴的图像检测、显式掩码、可观测文本沿用年龄与维修状态重建 |
| 不确定性 | 分位数区间、留出巡检事件校准与经验覆盖率报告 |
| API 可移植性 | MiniMax、OpenAI GPT、千问及自定义兼容服务；提取与解释接口分别配置 |
| 可核查实验 | 重复轨迹分组、六项消融、压力测试、配对组统计与明确的深度重建预算 |

各 Agent 采用明确的输入输出协议和固定工作流。预测专家为监督模型，LLM 负责有限的报告语义提取与证据解释；七个角色不意味着每条观察调用七次付费 API。

## 系统架构

![系统架构](docs/assets/architecture.svg)

调度 Agent 限定可用历史，四个专家处理数值状态、图像提取的曲线几何、报告特征与维护上下文。融合使用内层留出轨迹组上的专家预测学习融合关系。审计 Agent 展示质量标记，并可调用 LLM 解释证据，目标标签不会发送给解释接口。

特征协议、训练、区间校准、缓存与服务接口详见[技术手册](docs/TECHNICAL_MANUAL.zh-CN.md)。

## 仓库内的数据集

完整原始数据位于 [Multimodal data for prognostics/](Multimodal%20data%20for%20prognostics/)，保留上游 `Readme.docx`。

| 项目 | 本地副本审计结果 |
| --- | ---: |
| 设备目录 / 记录 | 50 / 7,626 |
| 有效 / 缺失曲线 | 6,088 / 1,538 |
| 规范化唯一报告 | 766 |
| 不同的数值、时间、文本轨迹组 | 30 |

设备 1–20 与 31–50 的输入历史分别相同，主要实验将每对设备放在同一分区。这是对提供副本的审计，不能据此确定原论文的数据实现。原始信号数组与真实报告生成时间不可用，曲线几何与文本连续沿用年龄作为派生观察处理。

来源与使用说明：[数据集说明](DATASET_NOTICE.md)、[LASAR 基准](https://www.lasar.polimi.it/?page_id=2970)、[关联论文](https://doi.org/10.1016/j.jmsy.2021.01.007)。

## 快速开始

已在 macOS ARM64、Python 3.14 下测试。可使用 Linux、macOS、WSL 或提供的 Docker 配置；当前未实现原生 Windows 文件锁，Docker 尚未在本机实测。

```bash
git clone https://github.com/13536309143/SteamWeave.git
cd SteamWeave
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m sg_agents.cli train --offline --method optimized
.venv/bin/python scripts/service.py start
```

工作台地址为 **http://127.0.0.1:8787**，API 文档为 **http://127.0.0.1:8787/docs**。默认离线流程使用仓库内固定的 MiniMax 语义资产。新的未缓存报告和实时解释需要配置 Key，模型文件在本地重新生成。

```bash
.venv/bin/python scripts/service.py restart
.venv/bin/python scripts/service.py stop
.venv/bin/python -m pytest -q
```

工作台支持观察记录选择、点预测与区间、缺失模态切换、历史曲线、自定义数值/文本/图像上传、CSV 导出、训练与可选择供应商的解释。

## 配置 API

将 [config/](config/) 中相应变量加入私有 `.env`。Key 仅保存在后端，不提交 Git。

| 供应商 | Key 环境变量 | 示例 | 验证状态 |
| --- | --- | --- | --- |
| MiniMax | `MINIMAX_API_KEY` | [MiniMax](config/minimax.env.example) | 已验证真实 API 请求 |
| OpenAI GPT | `OPENAI_API_KEY` | [OpenAI](config/openai.env.example) | 模拟协议与错误测试；真实验证需 Key |
| 千问 | `DASHSCOPE_API_KEY` | [千问](config/qwen.env.example) | 模拟协议与错误测试；真实验证需 Key |
| 自定义兼容服务 | `CUSTOM_API_KEY` | [自定义](config/custom.env.example) | 模拟协议与错误测试；需对具体部署实测 |

`LLM_PROVIDER` 选择语义提取接口，`LLM_AUDIT_PROVIDER` 可单独选择解释接口。支持 Chat Completions 与 Responses 格式，可配置模型、地址与输出 token 参数；适配器不会借用其他供应商的 Key。

```dotenv
LLM_PROVIDER=minimax
LLM_AUDIT_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_key
OPENAI_MODEL=your-gpt-model
```

```bash
.venv/bin/python -m sg_agents.cli check-api --provider openai
# 更换提取接口需要重建特征并重新训练。
.venv/bin/python -m sg_agents.cli train --provider qwen
```

仅更换解释供应商可保留预测模型；提取缓存按供应商、模型、地址和协议隔离。千问地区地址必须与 Key 匹配，详见[接口配置说明](docs/TECHNICAL_MANUAL.zh-CN.md#8-api-供应商凭证与缓存来源)。

## v2 优化结果

新增 **18 次外层评估**，每次在训练期内用三折比较八个固定候选。优化当前曲线形状编码，加入维修后重置的历史曲线与变化率，并让绝对误差模型参与选择。当前曲线缺失时使用原版增强专家融合。

| 场景 | 原版 MAE / adu | 优化 MAE / adu |
| --- | ---: | ---: |
| 轨迹分组五折 × 三种子 | 4.853 ± 0.405 | **4.089 ± 0.314** |
| 本地服务固定留出集 | 5.094 | **4.077** |
| 未来时段候选，三次运行 | **11.140** | 11.501 ± 0.029 |

分组 MAE 下降约 **15.7%**，本地服务留出误差下降约 **20.0%**。未来时段候选未改善，因此保留原版时间检查点。优化区间更宽：分组平均宽度 35.072 adu，原版为 28.818 adu。当前曲线全部屏蔽时，点预测与原版一致；此前已经观测到的历史仍可使用。

这是对已查看基准的探索性改进，可用性切换在首次缺失曲线诊断之后加入；不能视为新盲测或同条件超过论文的证明。候选与参数由训练内部选择，测试标签不进入预测特征。详见 [v2 优化报告](reports/optimization/OPTIMIZATION.zh-CN.md)、[逐折指标](reports/optimization/fold_metrics.csv)与[内层选择审计](reports/optimization/selection_audits.json)。

原版专家输出仍展示用于诊断。曲线可用时，最终预测使用新预测头；缺失时使用原版专家融合。使用 `python -m sg_agents.cli optimize --offline` 升级已有匹配检查点，或按上方快速开始从头训练。回到原版可运行 `python -m sg_agents.cli train --offline --method legacy`。

## 原版完整实验

**23 次外层评估 · 151 个模型评估 · 300 个压力条件 · 6 次深度拟合。** 分组 CV 采用五折、三个种子；六项消融和压力测试覆盖种子 17 的完整五折。深度结构重建采用一个种子。

| 模型 | 分组 CV MAE / adu | 评估次数 |
| --- | ---: | ---: |
| SteamWeave 完整系统 | **4.853 ± 0.405** | 15 |
| 直接特征融合 | 5.173 ± 0.423 | 15 |
| 专家固定平均 | 7.569 ± 0.740 | 15 |
| 数值 RF，100 棵树 | 7.441 ± 0.588 | 15 |
| 数值 SVR，C=100 | 6.688 ± 0.650 | 15 |
| 有限预算三分支 DNN 结构重建 | 13.914 ± 1.617 | 5 |

数值为折均值 ± 样本标准差，不是置信区间。深度基线种子数较少，采用公开说明的 80 轮上限与验证集早停。

![分组比较](reports/figures/model_comparison.png)

未来时段测试 **MAE 为 11.140 adu**，显示时间泛化差距。移除当前文本专家后，所测种子的误差下降；LLM 语义没有明确收益。架构角色不能证明预测价值。这些消融仍保留此前的时间与维护上下文，也未被用于悄悄替换生产配置。

[2021 年论文](https://doi.org/10.1016/j.jmsy.2021.01.007)在自身协议下报告 MAE 为 3.20 adu。本项目将其作为文献参照，在共享本地划分上运行公开训练选择的结构重建。数据与实现差异使其不能称为精确复现或跨论文优越性证明。按设备目录 ID 划分单独作为诊断，因为重复输入历史可能跨分区。

实验包括 0–100% 曲线/文本/双模态缺失、连续中断、报告延迟、维护及预测跨度分层、区间覆盖率、轨迹组 bootstrap 区间与采用 Holm 校正的配对消融。

| 产物 | 链接 |
| --- | --- |
| 双语实验报告 | [中文](reports/EXPERIMENTS.zh-CN.md) · [English](reports/EXPERIMENTS.md) |
| 固定协议与预算 | [协议](docs/EXPERIMENT_PROTOCOL.md) · [清单](reports/experiment_manifest.json) |
| 指标与统计 | [CSV](reports/fold_metrics.csv) · [汇总](reports/summary.json) · [配对分析](reports/paired_analysis.json) |
| 预测与划分 | [压缩 CSV](reports/predictions.csv.gz) · [划分清单](reports/run_manifests/) |
| 压力与深度训练 | [压力指标](reports/pressure_tests.json) · [逐轮历史](reports/neural_training_histories.json) |
| 可导出图表 | [PNG 与可编辑 SVG](reports/figures/) |

## 复现与扩展

```bash
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m research.run_suite
.venv/bin/python -m research.extend_stress
.venv/bin/python -m research.summarize
```

仓库包含仅覆盖当前词表的小型 GloVe 资产，`python -m research.glove` 可从官方来源重新生成。运行输出到已忽略的 `artifacts/research/`，与服务模型分开。仅在配置一致时恢复已完成运行；修改预算应使用新的 `--run-name`，详见[技术手册](docs/TECHNICAL_MANUAL.zh-CN.md#10-研究命令与比较边界)。

## 文档与引用

- 技术手册：[中文](docs/TECHNICAL_MANUAL.zh-CN.md) · [English](docs/TECHNICAL_MANUAL.md)。
- [贡献指南](CONTRIBUTING.md)、[安全说明](SECURITY.md)、[引用元数据](CITATION.cff)。
- [数据来源说明](DATASET_NOTICE.md)与 [GloVe/语义资产说明](research/assets/NOTICE.md)。

Yang, Z., Baraldi, P., & Zio, E. (2021). *A multi-branch deep neural network model for failure prognostics based on multimodal data*. Journal of Manufacturing Systems, 59, 42–50. [DOI](https://doi.org/10.1016/j.jmsy.2021.01.007)。

SteamWeave 研究包含三十个不同输入轨迹组的模拟基准。公开结果描述当前实现与数据副本，独立工业数据验证仍是后续工作。
