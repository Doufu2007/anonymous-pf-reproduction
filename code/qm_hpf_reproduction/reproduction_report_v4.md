# QM-HPF 独立重实现报告（权重尺度修正版）

运行目录：`runs-main-30-v4`。代码哈希：`312f3f5d4d9cac0d2758e5a7549ed7aa3fa6bffc832b49fb82ff622c82d89b36`。

旧版 `main-30-v3` 在低权粒子更新时混用了归一化权重和未归一化似然，不能作为论文证据。修正版统一使用原归一化权重乘以似然比，并加入恒等更新不改变相对权重的测试。

# Paired analysis

Run directory: `code/qm_hpf_reproduction/runs-main-30-v4`

## Aggregate RMSE

| Method | Mean | SD | 95% CI |
|---|---:|---:|---:|
| bootstrap | 0.9342 | 0.0586 | [0.9123, 0.9561] |
| mh | 0.9356 | 0.0625 | [0.9122, 0.9589] |
| qm_hpf | 0.9342 | 0.0630 | [0.9107, 0.9578] |
| rotation_only | 0.9439 | 0.0652 | [0.9195, 0.9682] |
| crossover_only | 0.9368 | 0.0616 | [0.9138, 0.9598] |

## Paired RMSE differences

Negative values favour QM-HPF.

| Comparison | Mean difference | 95% CI | Bootstrap CI | Cohen dz | Wins | Paired t p | Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qm_hpf_minus_bootstrap | 0.0000 | [-0.0061, 0.0061] | [-0.0059, 0.0056] | 0.000 | 16/30 | 0.9986 | 0.9193 | 0.9986 |
| qm_hpf_minus_mh | -0.0013 | [-0.0051, 0.0024] | [-0.0050, 0.0021] | -0.134 | 15/30 | 0.4683 | 0.6263 | 0.9367 |
| qm_hpf_minus_rotation_only | -0.0097 | [-0.0164, -0.0029] | [-0.0161, -0.0036] | -0.536 | 22/30 | 0.00642 | 0.004032 | 0.02568 |
| qm_hpf_minus_crossover_only | -0.0026 | [-0.0075, 0.0023] | [-0.0074, 0.0019] | -0.197 | 15/30 | 0.2888 | 0.5291 | 0.8664 |

The paired tests quantify uncertainty across the fixed seed set; they do not establish equivalence or hardware acceleration.


## 解释边界

该结果是基于公开方程和显式实现假设的独立重实现，不是严格复现。源论文代码及若干参数未公开；实验使用一维合成模型、30 个固定种子和经典模拟。与 Bootstrap PF 的均值差为约 `+0.000005`，95% CI 跨越零，因此不支持在该模型上宣称 QM-HPF 有精度优势。与 rotation-only 的差异只能说明当前组合实现不同，不能单独归因于量子旋转。
