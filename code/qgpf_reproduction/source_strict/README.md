# QGPF source-strict runner

该目录对应 Wei and Guo（2021）的 QGPF 源算法路线。上一级 `qgpf.py` 继续保留为 QGPF-like 独立重实现；本目录不会继承它的 `N=200`、精英比例或噪声解释。

源论文中可以直接锁定的设置包括：$N=400$、编码长度 $m=10$、最大遗传代数 $t=4$、常速度状态矩阵、量子染色体初始幅值 $(1/\sqrt2,1/\sqrt2)$ 以及量子旋转和观测流程。文中“比例 0.7”、噪声矩阵的单位、替换规则和 99 次实验的轨迹复用方式仍需源代码、原始数据或作者确认。

`run_source_strict.py` 默认读取 `source_config.json`。只要仍有 `UNRESOLVED` 或 `REQUEST_AUTHOR`，程序就会输出 `failure_artifacts/strict_blocked.json` 并停止。严格运行成功后，输出必须包含每个时间步和遗传代的染色体、振幅、测量位、旋转角、适应度、替换/保留标志、权重和 RNG 状态。

`source_kernel.py` 已实现源文中的常速度状态转移、径向观测、量子染色体初始化、测量、旋转门和代级 trace。替换/保留规则和噪声解释仍由严格配置提供，入口不会使用上级 QGPF-like 代码中的补设值。
