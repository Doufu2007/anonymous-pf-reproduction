# EC 概率解释变体修订验收

状态：**PASS，M2 可交付并放行后续 M3**。30 个数据集、每组 20 次更新全部保留，无种子排除或替换。统计单位为 30 个数据集，而非 600 次更新。

唯一算法变化是 `q=P(bit0)` 时 best 映射改为 `alpha*(1-D)+(1-alpha)*D`。`q<u` 观察 bit1、ellipse、正且与 subset 无关的 V、500 粒子/500 swarm、alpha=.8、c1=c2=.2、20 代/10 次 unchanged-best 停止、personal/global best 更新与 cyclic duplication 均保留。变体是**自洽解释变体**，不表示原作者实际意图或完整源算法实现。

## 主要结果

| 方法 | 每数据集 2D vector RMSE 均值 | SD | Median | Q25–Q75 |
|---|---:|---:|---:|---:|
| consistent | 4.0318749 | 0.3525490 | 4.0839754 | 3.8459643–4.2223109 |
| 历史 literal | 4.5504583 | 0.3931206 | 4.6025619 | 4.2918284–4.8221881 |
| exact singleton | 1.6147820 | 0.2341598 | 1.5691748 | 1.4667482–1.7307927 |
| same-fitness systematic | 4.7223624 | 0.4026258 | 4.7746855 | 4.5251392–4.9516802 |

| 预声明比较 | Paired mean difference | 20,000 whole-dataset bootstrap 95% CI | dz | Holm-3 sign-flip p |
|---|---:|---:|---:|---:|
| consistent−literal | −0.5185835 | [−0.5473334, −0.4902247] | −6.3746 | 0.0000600 |
| consistent−exact | +2.4170928 | [+2.2691638, +2.5476503] | +6.0674 | 0.0000600 |
| consistent−systematic | −0.6904875 | [−0.7230196, −0.6581399] | −7.4628 | 0.0000600 |

Bootstrap RNG 为 9301；50,000 次 sign-flip RNG 为 9302。三个 raw Monte Carlo p 值均为最小可报告值 `1/50001=0.0000199996`，不能写成 p=0；三项 Holm 校正值均为 `3/50001`。Descriptive paired-t 区间已另外保留在分析 JSON，不替换冻结的 bootstrap 主区间。

解释变体在这些已观察的归档合成输入上降低了 literal/systematic 的误差，但仍明显不及 exact。其 mean logfitness gap 从 literal 的 0.1854207 减至 0.1673074，distinct subset size 从 188.1300 减至 173.5683；mean executed generations 从 11.1700 增至 19.7933，每数据集 mean fitness evaluations 从 111,700 增至 197,933.3，binary-entry measurements 从 43.0320 million 增至 76.3683 million。目标搜索改善伴随更多实际工作，不能把相同最大代数称作相同实现成本。

最近 eligible singleton 的条件最优定理保持成立；该定理不推出 true-state RMSE 最优，也不否定完整 IMU tracker。此次结果仅支持这个明确选择模块合同中的概率解释影响，不识别一般量子收益。

## 执行与证据

- `runs-consistent-30-v2/`：完整 30 数据集，600 新更新，3 个比较者科学数组直接复用已核历史 raw。
- `replay-consistent-a/`、`replay-consistent-b/`：两个独立进程和目录的 seed-0 重放；9 项科学文件哈希一致，runtime 和时间戳排除。
- `analysis-consistent-30-v2/`：raw arrays 重算、paired RMSE/差值 CSV、分析 JSON、manifest。
- `readonly-verification.json`：990 项 manifest 输出文件、450 项历史输入记录、600 新更新和 1,800 复用记录全部核验；实际索引均值、truth RMSE、shared ellipse/V、fitness 精确上界、操作计数与 global-best 单调性全部 PASS。
- `test-red.log`、`test-direction-red.log`、`ecqpf_interpreted.red.txt`、`test-green.log`：真实先红后绿证据；最终 10 项 gold/property/contract 测试通过。

实际环境：Python 3.13.7、NumPy 2.4.2、Windows 11 build 26200、CPU float64；`PYTHONHASHSEED=0`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`，三次执行代码快照哈希一致。同 seed 一致只支持此环境，未验证跨硬件复现。

## 留存失败与修正

首次 `runs-consistent-30/` 在 seed-0 的环境信息采集阶段触发 `TypeError: show_config() got an unexpected keyword argument 'stream'`；`completed_summary_rows=[]`，未执行科学更新。该失败目录和 traceback 保留。修正仅把 NumPy build-config 捕获改为 `redirect_stdout`，未更改算子、参数、随机流或停止规则；10 项测试再次通过后，完整矩阵在新目录 `runs-consistent-30-v2/` 从 seed 0 重新执行。

所有本地原 manifest/config/snapshot 均保留。历史 runtime 与新 runtime 分别记录，不跨批排序；发布时路径净化应建立原文件与发布文件的映射并保留双边哈希，不覆盖本地运行事实。
