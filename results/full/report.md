# 结构感知统一原型：CPU仿真实验报告

本报告由 raw.jsonl 自动生成。所有输入均为构造输入；不含原始三项系统的复现结果、真实DiT模型或GPU加速结论。

环境：arm；macOS-26.6.2-arm64-arm-64bit；Python 3.12.14；NumPy 2.3.5。
输入随机种子：[7, 19, 31, 43, 59]；每项预热 2 次、计时 7 次。

计算代价是任务内的加权运算量，包含变换组织的计数开销；与实测毫秒数分列，未把运算量下降换算为耗时下降。
端到端时间包含任务适配、公共规划、方案校验、内核执行及训练参数更新；结果对照验证时间另存。
预算校准、初始索引构建和增量查询的初始缓存构建属于预处理，不计入对应在线时间；组合实验衡量已有缓存下的增量执行。

表中时间与加速比先在同一输入种子的重复运行中求均值，再在输入种子间求均值和样本标准差；小于1表示端到端减速。

|任务|输入情形|方案|验证通过|代价下降(%)|端到端加速比(均值±标准差)|端到端耗时(ms)|回退数|
|---|---|---|---:|---:|---:|---:|---:|
|dynamic|local_1|optimized|5/5|71.90|1.880±0.143|3.026|0|
|dynamic|local_1|reference|5/5|0.00|1.000±0.000|5.660|0|
|dynamic|local_8|optimized|5/5|50.10|1.277±0.230|4.901|0|
|dynamic|local_8|reference|5/5|0.00|1.000±0.000|6.162|0|
|dynamic|global_32|optimized|5/5|0.00|0.869±0.051|6.256|0|
|dynamic|global_32|reference|5/5|0.00|1.000±0.000|5.421|0|
|dynamic|cross_block|optimized|5/5|0.00|0.928±0.024|6.098|0|
|dynamic|cross_block|reference|5/5|0.00|1.000±0.000|5.662|0|
|logical|shared_8|optimized|5/5|76.12|1.902±0.022|33.591|0|
|logical|shared_8|reference|5/5|0.00|1.000±0.000|63.870|0|
|logical|shared_64|optimized|5/5|61.53|1.584±0.028|40.683|0|
|logical|shared_64|reference|5/5|0.00|1.000±0.000|64.423|0|
|logical|shared_384|optimized|5/5|16.24|1.069±0.018|60.353|0|
|logical|shared_384|reference|5/5|0.00|1.000±0.000|64.523|0|
|composition|cohorts_8_changed_0|optimized|5/5|93.72|1.336±0.133|3.118|0|
|composition|cohorts_8_changed_0|merge|5/5|76.14|1.189±0.100|3.491|0|
|composition|cohorts_8_changed_0|scope|5/5|93.72|2.278±0.461|1.867|0|
|composition|cohorts_8_changed_0|reference|5/5|0.00|1.000±0.000|4.146|0|
|composition|cohorts_8_changed_0|combined|5/5|93.66|1.497±0.283|2.845|0|
|composition|cohorts_8_changed_0.125|optimized|5/5|89.41|1.250±0.111|3.399|0|
|composition|cohorts_8_changed_0.125|merge|5/5|73.74|1.251±0.113|3.415|0|
|composition|cohorts_8_changed_0.125|scope|5/5|82.05|1.764±0.354|2.443|0|
|composition|cohorts_8_changed_0.125|reference|5/5|0.00|1.000±0.000|4.236|0|
|composition|cohorts_8_changed_0.125|combined|5/5|89.41|1.610±0.299|2.670|0|
|composition|cohorts_8_changed_0.5|optimized|5/5|82.63|1.131±0.149|3.943|0|
|composition|cohorts_8_changed_0.5|merge|5/5|72.65|1.361±0.162|3.280|0|
|composition|cohorts_8_changed_0.5|scope|5/5|46.84|1.326±0.079|3.349|0|
|composition|cohorts_8_changed_0.5|reference|5/5|0.00|1.000±0.000|4.433|0|
|composition|cohorts_8_changed_0.5|combined|5/5|82.63|1.522±0.265|2.956|0|
|composition|cohorts_8_changed_1|optimized|5/5|76.14|1.086±0.138|4.369|0|
|composition|cohorts_8_changed_1|merge|5/5|76.14|1.333±0.149|3.553|0|
|composition|cohorts_8_changed_1|scope|5/5|-0.04|0.958±0.070|4.955|0|
|composition|cohorts_8_changed_1|reference|5/5|0.00|1.000±0.000|4.713|0|
|composition|cohorts_8_changed_1|combined|5/5|76.11|1.432±0.211|3.325|0|
|composition|cohorts_128_changed_0|optimized|5/5|93.74|1.388±0.176|3.330|0|
|composition|cohorts_128_changed_0|merge|5/5|52.19|1.149±0.076|3.984|0|
|composition|cohorts_128_changed_0|scope|5/5|93.74|2.487±0.548|1.902|0|
|composition|cohorts_128_changed_0|reference|5/5|0.00|1.000±0.000|4.582|0|
|composition|cohorts_128_changed_0|combined|5/5|93.68|2.111±0.222|2.194|0|
|composition|cohorts_128_changed_0.125|optimized|5/5|82.18|1.241±0.164|3.782|0|
|composition|cohorts_128_changed_0.125|merge|5/5|45.14|1.062±0.127|4.420|0|
|composition|cohorts_128_changed_0.125|scope|5/5|82.00|2.115±0.329|2.237|0|
|composition|cohorts_128_changed_0.125|reference|5/5|0.00|1.000±0.000|4.652|0|
|composition|cohorts_128_changed_0.125|combined|5/5|82.18|1.772±0.273|2.658|0|
|composition|cohorts_128_changed_0.5|optimized|5/5|61.45|0.972±0.065|4.647|0|
|composition|cohorts_128_changed_0.5|merge|5/5|33.59|1.013±0.137|4.526|0|
|composition|cohorts_128_changed_0.5|scope|5/5|46.77|1.252±0.117|3.617|0|
|composition|cohorts_128_changed_0.5|reference|5/5|0.00|1.000±0.000|4.518|0|
|composition|cohorts_128_changed_0.5|combined|5/5|61.45|1.353±0.174|3.375|0|
|composition|cohorts_128_changed_1|optimized|5/5|52.19|0.944±0.088|5.197|0|
|composition|cohorts_128_changed_1|merge|5/5|52.19|1.260±0.189|3.926|0|
|composition|cohorts_128_changed_1|scope|5/5|-0.04|0.984±0.016|4.975|0|
|composition|cohorts_128_changed_1|reference|5/5|0.00|1.000±0.000|4.898|0|
|composition|cohorts_128_changed_1|combined|5/5|52.15|1.197±0.238|4.208|0|
|video|heterogeneous|reference|5/5|0.00|1.000±0.000|0.565|0|
|video|heterogeneous|uniform|5/5|6.70|0.969±0.025|0.583|0|
|video|heterogeneous|optimized|5/5|32.40|1.046±0.044|0.541|0|
|video|longer|reference|5/5|0.00|1.000±0.000|1.711|0|
|video|longer|uniform|5/5|11.73|1.052±0.032|1.626|0|
|video|longer|optimized|5/5|39.28|1.234±0.032|1.387|0|
|video|homogeneous|reference|5/5|0.00|1.000±0.000|0.567|0|
|video|homogeneous|uniform|5/5|10.06|1.009±0.028|0.563|0|
|video|homogeneous|optimized|5/5|11.45|0.999±0.010|0.568|0|
|video|strict|reference|5/5|0.00|1.000±0.000|0.549|0|
|video|strict|uniform|5/5|0.00|0.993±0.061|0.553|0|
|video|strict|optimized|5/5|0.00|0.982±0.052|0.559|0|

## 约束检查

- 动态图：逐项比较全部地标距离；小规模单元测试另用Floyd–Warshall独立验证。只实现单边界分块图上的精确距离维护，不实现FulBM的路径强弱标签。
- 逻辑查询训练：比较每轮前向结果、正负采样损失、实体及关系梯度，以及连续多轮更新后的参数。正样本来自构造知识图谱的集合语义求值，负样本取自答案补集。
- 视频张量去噪：在留出输入上比较全注意力参考输出，使用潜变量RMSE阈值；另记录跨帧差分RMSE。该指标不是VBench、视觉质量或真实视频生成模型精度。
- 组合实验：模型参数固定，依据输入变化传播缓存失效，再执行公共子表达式合并；任何参数快照变化都使相关缓存失效。训练时禁止跳过缓存分支的梯度。

## 可审计文件

raw.jsonl保留每种输入、方案、验证误差、候选执行方案、逐次计时样本和回退记录；inputs保存实际输入数组、表达式和图更新。
calibration.json记录视频预算、曲率代理、校准输入种子、校准耗时及校准误差。配置、源文件和输入文件均有SHA-256摘要。

公共规划器同时比较合法候选方案的加权代价；本原型尚未训练能准确预测不同CPU内核耗时的性能模型。因此，代价模型偏好的方案可能在小规模CPU运行中更慢，报告保留这些结果。
