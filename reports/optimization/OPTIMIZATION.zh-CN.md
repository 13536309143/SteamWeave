# SteamWeave v2 optimization / 优化报告

[English](OPTIMIZATION.md) · [中文](OPTIMIZATION.zh-CN.md)

这是在此前已查看的模拟基准上的探索性改进。八个候选方案在新外层预测前固定，三折内层验证仅使用外层训练记录选择特征、模型和损失。外层校准标签仅确定区间宽度，候选选择不接收外层测试标签。当前曲线可用性切换是在探索性的缺失曲线诊断后加入的，因此不是未经查看的测试集研究。

| Protocol / 协议 | Original MAE / 原 MAE | Optimized MAE / 优化 MAE | Runs / 运行 |
| --- | ---: | ---: | ---: |
| Grouped CV | 4.853 ± 0.405 | 4.089 ± 0.314 | 15 |
| Future periods | 11.140 ± 0.000 | 11.501 ± 0.029 | 3 |

分组折的 MSE 从 49.469 降至 39.073 adu²，MRE 从 8.742% 降至 7.036%。平均区间覆盖率从 94.6% 变为 97.2%，但区间更宽：28.818 → 35.072 adu。时间候选的 MSE 从 187.207 恶化为 222.680 adu²。报告保留这些权衡，不用 MAE 改善掩盖它们。

![Comparison](comparison.png)

改动：二次曲线形状描述；严格早于当前时点的同设备曲线状态及变化率；检测维修后重置历史；预测跨度交互；分组/前向内层选择；候选中包含绝对误差目标。语义特征作为可选候选参与比较，不强制进入每次预测。当前曲线可用时使用优化头，缺失时使用原版增强专家融合；切换取决于输入可用性，不取决于目标数值。

Paired group delta (optimized − original): **-0.741 adu**, bootstrap 95% **[-0.915, -0.573]**. Negative means improvement / 负值表示改善。

局限：没有新的独立数据集；仅三十个轨迹组；折标准差不是置信区间；时间运行使用相同固定划分；时间候选结果变差，因此服务保留原版未来时段检查点；不能据此声称在同条件下超过论文。历史不能恢复未观测维修或原始信号。压力测试仅屏蔽当前模态，保留此前可用曲线历史。

[Summary / 汇总](summary.json) · [Fold metrics / 折指标](fold_metrics.csv) · [Paired predictions / 配对预测](predictions.csv.gz) · [Inner selections / 内层选择](selection_audits.json) · [Pressure / 压力](pressure_tests.json) · [Manifest / 配置](manifest.json)

```bash
.venv/bin/python -m research.optimize run
.venv/bin/python -m research.optimize summarize
```
