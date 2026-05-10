# LEMS 消融实验文档

**创建日期**: 2026-05-10
**版本**: v1.0
**作者**: LEMS Project

---

## 1. 概述

消融实验（Ablation Study）是验证系统各模块贡献的重要方法。通过逐一禁用或替换特定模块，观察系统性能变化，从而量化各模块的贡献度。

本文档记录了LEMS系统消融实验的设计、实现和使用方法。

---

## 2. 实验目标

验证以下三个核心创新模块的贡献：

1. **AIR模块** - 基于CoT的环境与任务分析
2. **DREAM模块** - 诊断反思驱动的自适应进化算子
3. **多维度适应度评估** - 包括收敛过滤和降级选拔

---

## 3. 实验变体设计

### 3.1 单模块消融

| 变体名称 | 禁用模块 | 预期影响 |
|---------|---------|---------|
| `no_air` | AIR模块 | 初始代码质量下降，可能产生幻觉 |
| `no_dream` | DREAM模块 | 进化效率下降，搜索空间发散 |
| `no_convergence_filter` | 收敛过滤 | 虚假收敛候选被误选 |
| `no_degradation_selection` | 降级选拔 | 有效候选被错误淘汰 |
| `single_fitness` | 多维适应度 | 仅使用成功率，忽略其他指标 |

### 3.2 组合消融

| 变体名称 | 禁用模块组合 | 预期影响 |
|---------|------------|---------|
| `no_air_no_dream` | AIR + DREAM | 初始质量差且进化效率低 |
| `no_air_no_fitness` | AIR + 多维适应度 | 初始质量差且评估不全面 |
| `no_dream_no_fitness` | DREAM + 多维适应度 | 进化效率低且评估不全面 |

### 3.3 基线

| 变体名称 | 说明 |
|---------|------|
| `full` | 完整LEMS系统，作为性能基线 |

---

## 4. 实现细节

### 4.1 文件位置

```
LEMS/
├── ablation_study.py          # 消融实验主脚本
└── docs/
    └── ABLATION_STUDY.md      # 本文档
```

### 4.2 核心类设计

#### AblationVariant（基类）

```python
class AblationVariant:
    """消融实验变体基类"""

    def modify_config(self, config: dict) -> dict:
        """修改配置以禁用特定模块"""
        return config

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """修改agent行为以禁用特定模块"""
        pass

    def get_generation_result(self, agent, generation, use_real_training) -> dict:
        """获取一代进化结果，可覆盖默认行为"""
        return agent.step(generation=generation, use_real_training=use_real_training)
```

#### 各变体实现方式

**NoAIR（禁用AIR模块）**
- 跳过阶段一的CoT分析
- 直接使用标准prompt生成候选代码
- 设置 `agent._skip_cot_analysis = True`

**NoDream（禁用DREAM模块）**
- 强制使用静态循环分配算子（F1→F2→F3→L1）
- 忽略LLM的算子选择
- 覆盖 `_apply_dream_mutation` 方法

**NoConvergenceFilter（禁用收敛过滤）**
- 跳过三重收敛判别（F_mean, F_std, F_slope）
- 直接取fitness最高的候选
- 简化 `analyze_results` 方法

**NoDegradationSelection（禁用降级选拔）**
- 只选择收敛的候选
- 未收敛候选直接被淘汰
- 修改 `analyze_results` 的选拔逻辑

**SingleDimensionFitness（单维适应度）**
- 将fitness权重配置修改为仅保留成功率
- 其他维度权重设为0

### 4.3 AblationStudy类

```python
class AblationStudy:
    """消融实验管理器"""

    AVAILABLE_VARIANTS = {
        'full': (FullSystem, "完整LEMS系统"),
        'no_air': (NoAIR, "禁用AIR模块"),
        'no_dream': (NoDream, "禁用DREAM模块"),
        # ... 其他变体
    }

    def run_variant(self, variant_name, num_generations, episode_num, ...) -> dict:
        """运行单个消融实验变体"""

    def run_full_study(self, num_generations, ...) -> dict:
        """运行完整消融实验"""

    def save_results(self, output_dir):
        """保存实验结果"""

    def print_summary(self):
        """打印实验结果摘要"""
```

---

## 5. 使用方法

### 5.1 基本用法

```bash
# 查看可用变体
python ablation_study.py --list-variants

# 运行完整消融实验（所有变体）
python ablation_study.py --num_generations 5 --episode_num 100

# 运行特定变体
python ablation_study.py --variants full no_air no_dream --num_generations 3

# 使用模拟训练快速测试
python ablation_study.py --num_generations 3 --no-real-training
```

### 5.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `llm_reward_agent/config/llm_config.yaml` | LLM配置文件路径 |
| `--num_generations` | 5 | 进化代数 |
| `--episode_num` | 100 | 每个候选的训练回合数 |
| `--use_real_training` | True | 使用真实训练 |
| `--no-real-training` | - | 使用模拟训练（快速测试） |
| `--env_file` | `MADDPG/envs/simple_tag_env.py` | 环境文件路径 |
| `--variants` | 全部 | 要运行的变体列表 |
| `--output_dir` | `experiments/ablation_study` | 结果输出目录 |

### 5.3 运行示例

**完整实验（推荐用于论文）**

```bash
python ablation_study.py \
    --num_generations 10 \
    --episode_num 200 \
    --output_dir experiments/ablation_study_full
```

**快速验证（开发调试）**

```bash
python ablation_study.py \
    --variants full no_air no_dream \
    --num_generations 3 \
    --no-real-training
```

---

## 6. 输出结果

### 6.1 目录结构

```
experiments/ablation_study/
├── ablation_results.json      # 详细JSON结果
└── ablation_summary.txt       # 可读的摘要报告
```

### 6.2 JSON结果格式

```json
{
  "full": {
    "variant": "full",
    "description": "完整LEMS系统",
    "num_generations": 5,
    "total_time": 1234.5,
    "best_fitness": 0.85,
    "final_fitness": 0.82,
    "avg_fitness": 0.78,
    "fitness_history": [0.65, 0.72, 0.78, 0.82, 0.85],
    "generation_results": [...]
  },
  "no_air": {
    // ...
  }
}
```

### 6.3 摘要报告示例

```
LEMS 消融实验结果摘要
================================================================================

变体                      最优Fitness     最终Fitness
-------------------------------------------------------
full                      0.8523          0.8234
no_dream                  0.7845          0.7623
no_convergence_filter     0.7234          0.7012
no_air                    0.6523          0.6345
single_fitness            0.6123          0.5923
no_degradation_selection  0.5823          0.5623

各模块贡献分析:
完整系统最优Fitness: 0.8523

  no_air                    性能下降:  23.48%
  no_dream                  性能下降:   7.95%
  no_convergence_filter     性能下降:  15.12%
  single_fitness            性能下降:  28.16%
  no_degradation_selection  性能下降:  31.68%
```

---

## 7. 结果分析指南

### 7.1 贡献度计算

```python
# 各模块贡献度计算公式
contribution = (full_fitness - variant_fitness) / full_fitness * 100%

# 示例
full_best = 0.85
no_air_best = 0.65
air_contribution = (0.85 - 0.65) / 0.85 * 100% = 23.5%
```

### 7.2 预期结果

根据理论分析，各模块贡献度预期排序：

1. **AIR模块** - 贡献度最高（解决初始失真问题）
2. **多维适应度** - 贡献度较高（提供准确评估）
3. **DREAM模块** - 贡献度中等（提升进化效率）
4. **收敛过滤** - 贡献度中等（避免虚假收敛）
5. **降级选拔** - 贡献度较低（处理极端情况）

### 7.3 可视化建议

使用 `visualization/evolution_plot.py` 绘制：

1. **进化曲线对比图** - 各变体fitness随代数变化
2. **最终性能柱状图** - 各变体最终fitness对比
3. **贡献度饼图** - 各模块贡献占比

---

## 8. 注意事项

### 8.1 实验设置

- **随机种子**：建议设置固定种子以保证可重复性
- **重复实验**：建议每个变体运行3-5次取平均值
- **训练回合数**：建议使用200+回合以获得稳定结果

### 8.2 资源消耗

- **单个变体**：约30-60分钟（5代，100回合）
- **完整实验**：约5-10小时（9个变体）
- **建议**：先用模拟训练验证，再用真实训练获取最终结果

### 8.3 常见问题

**Q: 为什么某些变体性能反而更好？**
A: 可能原因：
- 随机性导致的波动（增加重复实验次数）
- 该模块在特定场景下产生负面影响
- 训练回合数不足，结果不稳定

**Q: 如何判断模块贡献是否显著？**
A: 建议进行统计显著性检验（如t检验），p<0.05认为显著。

---

## 9. 扩展实验

### 9.1 参数敏感性分析

可扩展消融实验以分析参数敏感性：

```python
# 示例：分析v_th参数对收敛过滤的影响
class VThSensitivity(AblationVariant):
    def __init__(self, v_th: float):
        super().__init__(f"v_th_{v_th}", f"v_th={v_th}")
        self.v_th = v_th

    def modify_config(self, config):
        config['fitness']['convergence']['v_th'] = self.v_th
        return config
```

### 9.2 组合效应分析

分析模块组合的协同效应：

```python
# 协同效应 = 完整系统性能 - (各模块独立贡献之和)
synergy = full_fitness - sum(individual_contributions)
```

---

## 10. 参考文献

1. EUREKA: Human-Level Reward Design via Coding Large Language Models
2. LEMS项目核心创新模块技术解析（工作4.md）
3. LEMS系统架构文档

---

## 11. 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-05-10 | v1.0 | 初始版本，实现9个消融变体 |

---

*文档生成时间：2026年5月10日*
*项目版本：LEMS v2.0*
