# 结构感知高效推理工程原型

本仓库将论文的形式化框架实现为可运行的公共规划器、三个领域适配器和组合变换实验。三个适配器使用同一套任务结构、计算范围、节点分组、预算配置、方案合法性检查和执行入口。

这是**构造输入上的CPU仿真原型**。动态图维护真实计算地标距离；知识图谱任务真实计算查询训练损失、反向梯度与参数更新；视频任务真实执行合成时空张量的块注意力与迭代去噪。它验证三类结构感知机制和公共执行框架，不复现原始FulBM、FlashEKGR和PASA的全部算法、模型或GPU实现。

## 运行

依赖Python 3.10及以上、NumPy 1.24–2.x，不需要GPU、模型下载或外部数据集。在仓库目录执行：

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m structure_aware.experiments --config configs/quick.json --output results/my_quick_run
python3 -m structure_aware.audit results/my_quick_run
```

完整实验：

```bash
python3 -m structure_aware.experiments --config configs/full.json --output results/my_full_run
python3 -m structure_aware.audit results/my_full_run
```

已有NumPy时不必安装本项目，在仓库目录直接运行上述模块即可。输出目录必须是新的，避免覆盖既有原始实验记录。完整配置使用5个输入种子、每种条件预热2次并计时7次；视频校准使用另外3个种子。

## 仓库内容

|位置|用途|
|---|---|
|`structure_aware/core.py`|公共结构表示、范围收缩、公共子表达式合并、边际收益预算分配、代价选择、方案检查与回退|
|`structure_aware/adapters/dynamic.py`|单边界分块动态图上的精确地标距离维护|
|`structure_aware/adapters/logical.py`|构造知识图谱、复杂逻辑查询、训练自动微分、版本缓存|
|`structure_aware/adapters/video.py`|合成视频潜变量、块注意力、去噪轨迹、离线校准和留出质量验证|
|`structure_aware/experiments.py`|三个任务及组合实验的公共计时、验证、输入保存和原始记录|
|`structure_aware/report.py`|从原始记录生成Markdown、CSV、JSON及论文结果表|
|`structure_aware/audit.py`|检查配置覆盖、源文件及输入摘要、计时样本、误差阈值和汇总一致性|
|`tests/`|独立最短路核验、数值梯度检查、缓存失效、变换组合、预算约束及回退测试|
|`examples/`|四个小规模任务的完整依赖图、候选方案、实际节点范围和合并分组，可直接查看JSON|
|`tools/export_plans.py`|重新导出上述可检查的执行方案|
|`docs/design.md`|接口、算法、约束、代价口径和实验设计|
|`results/full/`|当前代码的完整实验记录及报告|
|`results/pilot_full/`|优化公共规划器前的诊断实验及对应源文件快照|
|`results/quick/`|当前代码的小配置检查|

## 统一规划如何落实

1. 适配器生成`Task`，描述计算单元及依赖，并提供合法的处理范围或校准预算。
2. `Planner`通过公共变换生成候选`Plan`：完整执行、范围收缩、节点合并、二者组合，或预算调整。
3. 公共代价评估比较候选方案，保留获益较高的合法方案。结构等价分析在同一任务的候选之间复用。
4. `Runtime`检查方案并调用领域后端。实验框架记录实际执行计数、准备/规划/执行耗时和结果对照误差。
5. 留出结果不满足质量阈值时，实验框架回退至完整执行，并保留首次失败和回退开销。这里使用仿真参考结果作为核验依据。

组合实验使用同一逻辑查询图：部分输入改变后，经公共依赖传播确定需要重算的范围，再在该范围内做公共子表达式合并。分别测试完整重算、仅收缩、仅合并、组合执行和自动选案。模型参数固定；参数变化会使缓存失效；训练时不允许通过跨批次缓存跳过梯度贡献。

## 阅读实验结果

先读`results/full/report.md`，再查`summary.csv`和`raw.jsonl`。`raw.jsonl`保留每个输入种子、候选方案、实际选案、误差和每次计时样本。`inputs/`保存实际输入数组、图更新与查询表达式。`calibration.json`保存视频预算校准记录。

加权运算量与实测耗时独立报告。公共规划器按加权运算量选案，尚未使用能准确预测CPU耗时的模型，因此小任务中可能出现计算量下降而端到端减速。所有这类结果均保留。视频RMSE衡量合成潜变量与全注意力参考输出的差异，不能等同真实视频模型的生成质量。

源文件随每次实验保存到`source_snapshot/`。当前代码的实验用常规审计命令；历史实验用：

```bash
python3 -m structure_aware.audit results/pilot_full --archived
```

完整设计见`docs/design.md`。本次论文补充的修改说明及可应用的源文件差异分别保存于`docs/论文修改说明.md`与`docs/thesis-integration.patch`；结果表由论文绪论直接引用，避免手工抄录实验数值。

论文结果表`thesis_results.tex`由原始记录自动生成，禁止手动修改数值。输入文件摘要、配置摘要、代码摘要及每次运行的状态保存在对应输出目录中。
