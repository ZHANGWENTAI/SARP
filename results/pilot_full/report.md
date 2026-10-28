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
|dynamic|local_1|optimized|5/5|71.90|1.660±0.109|5.478|0|
|dynamic|local_1|reference|5/5|0.00|1.000±0.000|9.089|0|
|dynamic|local_8|optimized|5/5|50.10|1.280±0.127|8.499|0|
|dynamic|local_8|reference|5/5|0.00|1.000±0.000|10.826|0|
|dynamic|global_32|optimized|5/5|0.00|0.984±0.129|15.149|0|
|dynamic|global_32|reference|5/5|0.00|1.000±0.000|14.761|0|
|dynamic|cross_block|optimized|5/5|0.00|0.901±0.097|24.170|0|
|dynamic|cross_block|reference|5/5|0.00|1.000±0.000|21.696|0|
|logical|shared_8|optimized|5/5|76.13|1.395±0.086|92.361|0|
|logical|shared_8|reference|5/5|0.00|1.000±0.000|128.117|0|
|logical|shared_64|optimized|5/5|61.54|1.216±0.241|59.634|0|
|logical|shared_64|reference|5/5|0.00|1.000±0.000|68.022|0|
|logical|shared_384|optimized|5/5|16.25|0.758±0.141|195.539|0|
|logical|shared_384|reference|5/5|0.00|1.000±0.000|141.138|0|
|composition|cohorts_8_changed_0|optimized|5/5|93.72|0.969±0.369|18.008|0|
|composition|cohorts_8_changed_0|merge|5/5|76.18|1.074±0.269|15.955|0|
|composition|cohorts_8_changed_0|scope|5/5|93.72|2.439±0.667|7.149|0|
|composition|cohorts_8_changed_0|reference|5/5|0.00|1.000±0.000|17.084|0|
|composition|cohorts_8_changed_0|combined|5/5|93.66|1.376±0.334|12.507|0|
|composition|cohorts_8_changed_0.125|optimized|5/5|89.41|0.725±0.100|14.992|0|
|composition|cohorts_8_changed_0.125|merge|5/5|73.77|0.917±0.154|11.803|0|
|composition|cohorts_8_changed_0.125|scope|5/5|82.05|1.601±0.444|6.875|0|
|composition|cohorts_8_changed_0.125|reference|5/5|0.00|1.000±0.000|10.719|0|
|composition|cohorts_8_changed_0.125|combined|5/5|89.41|1.037±0.168|10.494|0|
|composition|cohorts_8_changed_0.5|optimized|5/5|82.64|0.732±0.039|15.119|0|
|composition|cohorts_8_changed_0.5|merge|5/5|72.69|0.947±0.026|11.699|0|
|composition|cohorts_8_changed_0.5|scope|5/5|46.84|1.629±0.078|6.788|0|
|composition|cohorts_8_changed_0.5|reference|5/5|0.00|1.000±0.000|11.073|0|
|composition|cohorts_8_changed_0.5|combined|5/5|82.64|1.117±0.132|9.960|0|
|composition|cohorts_8_changed_1|optimized|5/5|76.18|0.635±0.053|14.808|0|
|composition|cohorts_8_changed_1|merge|5/5|76.18|0.949±0.111|9.956|0|
|composition|cohorts_8_changed_1|scope|5/5|-0.04|0.955±0.145|9.940|0|
|composition|cohorts_8_changed_1|reference|5/5|0.00|1.000±0.000|9.422|0|
|composition|cohorts_8_changed_1|combined|5/5|76.14|0.949±0.102|9.941|0|
|composition|cohorts_128_changed_0|optimized|5/5|93.74|0.830±0.065|9.424|0|
|composition|cohorts_128_changed_0|merge|5/5|52.22|0.844±0.061|9.269|0|
|composition|cohorts_128_changed_0|scope|5/5|93.74|2.191±0.602|3.732|0|
|composition|cohorts_128_changed_0|reference|5/5|0.00|1.000±0.000|7.800|0|
|composition|cohorts_128_changed_0|combined|5/5|93.68|1.189±0.103|6.585|0|
|composition|cohorts_128_changed_0.125|optimized|5/5|82.19|0.714±0.044|9.877|0|
|composition|cohorts_128_changed_0.125|merge|5/5|45.17|0.829±0.067|8.521|0|
|composition|cohorts_128_changed_0.125|scope|5/5|82.00|1.822±0.255|3.924|0|
|composition|cohorts_128_changed_0.125|reference|5/5|0.00|1.000±0.000|7.053|0|
|composition|cohorts_128_changed_0.125|combined|5/5|82.19|1.132±0.093|6.253|0|
|composition|cohorts_128_changed_0.5|optimized|5/5|61.47|0.654±0.090|13.054|0|
|composition|cohorts_128_changed_0.5|merge|5/5|33.62|0.798±0.100|10.676|0|
|composition|cohorts_128_changed_0.5|scope|5/5|46.77|1.432±0.191|5.964|0|
|composition|cohorts_128_changed_0.5|reference|5/5|0.00|1.000±0.000|8.475|0|
|composition|cohorts_128_changed_0.5|combined|5/5|61.47|0.983±0.141|8.693|0|
|composition|cohorts_128_changed_1|optimized|5/5|52.22|0.581±0.048|16.102|0|
|composition|cohorts_128_changed_1|merge|5/5|52.22|0.838±0.073|11.173|0|
|composition|cohorts_128_changed_1|scope|5/5|-0.04|0.994±0.137|9.495|0|
|composition|cohorts_128_changed_1|reference|5/5|0.00|1.000±0.000|9.324|0|
|composition|cohorts_128_changed_1|combined|5/5|52.18|0.830±0.063|11.262|0|
|video|heterogeneous|reference|5/5|0.00|1.000±0.000|1.185|0|
|video|heterogeneous|uniform|5/5|6.70|0.995±0.007|1.191|0|
|video|heterogeneous|optimized|5/5|32.40|1.072±0.012|1.105|0|
|video|longer|reference|5/5|0.00|1.000±0.000|3.597|0|
|video|longer|uniform|5/5|11.73|1.019±0.023|3.532|0|
|video|longer|optimized|5/5|39.28|1.245±0.020|2.890|0|
|video|homogeneous|reference|5/5|0.00|1.000±0.000|1.180|0|
|video|homogeneous|uniform|5/5|10.06|0.996±0.003|1.185|0|
|video|homogeneous|optimized|5/5|11.45|0.988±0.011|1.195|0|
|video|strict|reference|5/5|0.00|1.000±0.000|1.095|0|
|video|strict|uniform|5/5|0.00|1.034±0.053|1.059|0|
|video|strict|optimized|5/5|0.00|0.996±0.009|1.100|0|

## 约束检查

- 动态图：逐项比较全部地标距离；小规模单元测试另用Floyd–Warshall独立验证。只实现单边界分块图上的精确距离维护，不实现FulBM的路径强弱标签。
- 逻辑查询训练：比较每轮前向结果、正负采样损失、实体及关系梯度，以及连续多轮更新后的参数。正样本来自构造知识图谱的集合语义求值，负样本取自答案补集。
- 视频张量去噪：在留出输入上比较全注意力参考输出，使用潜变量RMSE阈值；另记录跨帧差分RMSE。该指标不是VBench、视觉质量或真实视频生成模型精度。
- 组合实验：模型参数固定，依据输入变化传播缓存失效，再执行公共子表达式合并；任何参数快照变化都使相关缓存失效。训练时禁止跳过缓存分支的梯度。

## 可审计文件

raw.jsonl保留每种输入、方案、验证误差、候选执行方案、逐次计时样本和回退记录；inputs保存实际输入数组、表达式和图更新。
calibration.json记录视频预算、曲率代理、校准输入种子、校准耗时及校准误差。配置、源文件和输入文件均有SHA-256摘要。

公共规划器同时比较合法候选方案的加权代价；本原型尚未训练能准确预测不同CPU内核耗时的性能模型。因此，代价模型偏好的方案可能在小规模CPU运行中更慢，报告保留这些结果。
