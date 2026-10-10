# MR.CLAM 外部数据审计

本目录实现的是 MR.CLAM 真实传感器输入上的**声明式审计合同**。它不是
QM-HPF、QGPF 或 EC-QPF 源算法的完整复现。算法只接收里程计、地标地图和
range-bearing 观测；`Groundtruth.dat` 只由 `run_mrclam.py` 的评估器读取，
不会进入粒子初始化、状态转移或似然计算。

## 方法标签

- `B-stable`：Bootstrap PF；每个事件使用 `budget` 个经典粒子、系统重采样。
- `RM-stable`：Bootstrap PF 后进行 resample-move；为保持事件级 likelihood 上限，
  粒子数为 `budget/2`，候选移动使用声明的转移先验。
- `Q-stable`：QGPF-like 量子启发式选择模块；二值编码、旋转、测量和似然接受，
  但不声称等于任何源论文的完整算法。
- `Q-stable-retain`：与 `Q-stable` 相同的编码和预算，保留候选适应度改进；它是
  机制对照，不是源算法名称。

所有方法都执行相同的外部输入窗口和地图边界，且每个事件的
`likelihood_evals` 预算由运行时 guardrail 检查。算法之间的运行时间不能由
likelihood 计数直接推出。

## 计数定义

`analyze_results.py` 输出的 `audit_counts.csv` 是每个 dataset/robot/method
跨种子的计数总和，同时提供 `mean_per_seed_*` 描述每次 Monte-Carlo 审计的平均计数；
它不会把这些种子当成独立真实轨迹。

| 计数 | 定义 |
|---|---|
| `likelihood_evals` | 完整状态向量的一次批量观测似然评估；主要预算单位 |
| `scalar_observation_evals` | 状态与单个 range-bearing 观测对的评估数 |
| `candidate_evals` | Q/RM 候选状态的似然评估调用数 |
| `proposal_candidate_attempts` | 提议候选被生成并送入接受规则的次数 |
| `proposal_chromosome_events` | Q 选择模块进入染色体/候选更新的粒子数 |
| `rotation_calls` | 对不匹配的量子概率条目执行旋转的次数 |
| `crossover_operations` | 交叉算子次数；当前合同未使用时应为 0 |
| `measurement_bit_observations` | 二值概率测量的 bit 数 |
| `probability_entry_updates` | 概率向量条目被更新的次数 |
| `fitness_evals` | 候选适应度（当前为观测似然）评估次数 |
| `accepts` / `rejects` | 候选接受/拒绝次数 |
| `resampling_calls` | 系统重采样调用次数 |
| `motion_state_segment_evals` | 状态粒子与分段里程计控制的推进次数 |

不同计数的单位不同，不能相加为“总计算量”。`runtime_s` 还受 Python、硬件和
线程环境影响，因此报告中只作为次要描述性指标。

## 分析与证据边界

运行：

```powershell
python analyze_results.py --runs ../../../../local-data/mrclam/runs-main-30 `
  --expected-runs 240 --out ../../../../local-data/mrclam/analysis-main-30
```

脚本会：

1. 检查 240 个 run manifest、manifest 文件哈希和 `predictions.csv` 的端点重算；
2. 输出 pre-move 与 post-assimilation 径向位置 RMSE 的均值、SD、bootstrap CI；
3. 在每个 dataset/robot 内按相同 seed 与 `B-stable` 配对，输出方法差值、条件
   Monte-Carlo bootstrap CI 和描述性 `dz_mc`；不输出 p 值或六重显著性结论；
4. 输出 ESS fraction、distinct-particle count、runtime 范围及审计计数；
5. 若数据齐全生成 `category4-journal-draft/figs/mrclam-realdata-audit.pdf`。

bootstrap CI 只描述固定 MR.CLAM 序列上的 Monte-Carlo 种子不确定性，不能写成
跨机器人、跨场景或总体数据集泛化证据。图注必须说明同一真实轨迹上的种子不是
独立轨迹。原始 MR.CLAM 文件保存在仓库外，分析产物不复制第三方观测或真值数组。

```powershell
python -m unittest test_algorithm test_data test_analysis -v
```
