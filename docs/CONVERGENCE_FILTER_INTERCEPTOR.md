# 任务二：多维收敛过滤拦截器

> 文档版本：v1.1
> 完成日期：2026-04-09
> 对应模块：`llm_reward_agent/tools/log_analyzer.py`、`llm_reward_agent/agent/reward_design_agent.py`

---

## 1. 背景与问题

在非平稳的多智能体协同任务（如 predator-prey 围捕场景）中，盲目取最终或最高适应度（Fitness）的策略存在根本性缺陷：高方差的"虚假收敛"（spurious convergence）会欺骗系统。典型表现为训练曲线的最终回报数值看起来很高，但实际上：

- 曲线剧烈震荡，方差未收敛
- 早期待评估均值与晚期待评估均值无显著提升
- 全局斜率趋势方向不确定

为此，实装了严格的三重收敛过滤拦截器，对所有候选奖励函数进行数学级时序诊断。

---

## 2. 核心算法

### 2.1 论文参照

算法严格对照 EUREKA 论文中的式(5)、式(6)、式(7)设计：

| 判别式 | 含义 | 数学定义 |
|--------|------|----------|
| **F_mean**（式5）| 增益判断 | J_lm > J_em（晚期待均值 > 早期待均值） |
| **F_std**（式6）| 收敛判断（方差收缩） | (J_lv / J_ev) < v_th |
| **F_slope**（式7）| 趋势判断（正向斜率） | cov(t, fitness) > 0 |

### 2.2 参数说明

在 `fitness_config.convergence` 中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | **150** | 早期待评估窗口大小（回合数）。建议值 = 总回合数 × 5%~15% |
| `beta` | **300** | 晚期待评估窗口大小（回合数）。建议值 = 总回合数 × 10%~15% |
| `v_th` | **0.8** | 方差收缩容忍阈值（比值）。**已从 0.5 调整为 0.8**，详见下方说明 |

> **阈值调整说明（v_th: 0.5 → 0.8）**
>
> 多智能体强化学习环境（如 predator-prey 围捕场景）相比单智能体环境具有更高的非平稳性和噪声，主要体现在：
> - 各智能体策略同时演化，环境始终处于动态变化中
> - 协作战术需要更长的探索周期，策略方差收敛更慢
> - 追捕者与逃逸者的博弈存在天然的非平稳性
>
> 因此，将 `v_th` 从严格的 0.5 调整为更宽松的 0.8，在保证基本收敛趋势的前提下：
> - **避免过度拦截**：防止有潜力的候选因方差收缩不足而被误杀
> - **提升进化效率**：让更多候选进入下一代，促进协同行为的涌现
> - **适配多智能体特性**：方差波动是多智能体系统的固有特征，不应视为收敛失败
>
> **窗口大小调整说明（alpha/beta: 20/40 → 150/300）**
>
> 窗口大小的选择直接影响收敛判定的稳健性：
> - `alpha` 偏小（如 20）：仅用极少量回合的均值代表"早期"，极易受初始随机性干扰，统计不稳定
> - `beta` 偏小（如 40）：仅用极少量回合的均值代表"末期"，容易被尾部噪声误导
> - 两者偏小都可能导致收敛判定在边界情况下剧烈波动
>
> 推荐按训练总回合数按比例设置（5%~15%），示例：
>
> | 训练回合数 | alpha（5%） | beta（10%） |
> |-----------|-------------|-------------|
> | 1000 | 50 | 100 |
> | 2000 | 100 | 200 |
> | **3000** | **150** | **300** |
> | 5000 | 250 | 500 |
>
> 参考值：
> - 单智能体（短训练 <1000 回合）：`alpha=20, beta=40, v_th=0.5`
> - **多智能体（长训练 >=3000 回合）：`alpha=150, beta=300, v_th=0.8`**（当前默认值）

- 当 episode 总数 < 10 时，默认放行（数据量不足以判断）
- 窗口大小自动防越界：`alpha = min(alpha, L // 3)`

### 2.3 算法流程

```
输入：episode_fitnesses [f_1, f_2, ..., f_L]

1. 若 L < 10 → 直接放行（is_converged = True）

2. 早期待 E[:alpha]，晚期待 E[-beta:]
   - J_em = mean(E[:alpha])
   - J_lm = mean(E[-beta:])
   - J_ev = std(E[:alpha])
   - J_lv = std(E[-beta:])

3. 全局斜率协方差：
   - cov = cov(t, episode_fitnesses)

4. 三重判别：
   - f_mean  = (J_lm > J_em)
   - f_std   = ((J_lv / (J_ev + 1e-8)) < v_th)
   - f_slope = (cov > 0)

5. 完全收敛 = f_mean AND f_std AND f_slope
```

---

## 3. 模块改动

### 3.1 `log_analyzer.py`

**新增 `convergence` 配置块**（`__init__`）：

```python
self.fitness_config = fitness_config or {
    'weights': {...},
    'normalize': {...},
    'convergence': {
        'alpha': 20,   # 早期待评估窗口
        'beta': 40,    # 晚期待评估窗口
        'v_th': 0.5    # 方差收缩容忍阈值
    }
}
```

**新增 `_evaluate_convergence(episode_fitnesses)` 方法**：
- 核心拦截逻辑，严格实现论文三重判别
- 返回 `f_mean`、`f_std`、`f_slope`、`is_converged` 及 `details` 诊断字符串

**重写 `parse_logs()` 方法**：
- 从 `training_log.json` 提取每回合 `total_reward` 作为时序序列
- 在计算 `fitness` 后调用 `_evaluate_convergence` 执行拦截
- 若未提取到时序数据，打印警告并默认放行
- 将 `convergence_status` 注入 `metrics` 字典返回

**更新 `generate_analysis_report()` 方法**：
- 在报告末尾新增「收敛拦截诊断」节，展示三重判别结果

### 3.2 `reward_design_agent.py`

**重构 `analyze_results()` — 降级选拔算法**：

| 优先级 | 触发条件 | 选拔策略 |
|--------|----------|----------|
| **优先级 1** | 存在完全收敛候选（`is_converged == True`） | 取 fitness 最高者 |
| **优先级 2** | 无完全收敛，但存在趋势上升候选（`f_mean AND f_slope`） | 放宽 f_std 要求，取 fitness 最高者 |
| **优先级 3** | 全军覆没（无任何候选收敛） | 降级取绝对 fitness 最高者，进行强制突变 |

**重构 `_simulate_training()` — 注入模拟收敛标记**：
- 随机生成 `is_converged` 及三重判别结果
- 确保模拟数据能覆盖所有三种选拔优先级场景
- 输出中打印收敛状态供调试观察

---

## 4. 数据流

```
训练沙盒
    │
    ▼
training_log.json
    │  (含 episodes[].total_reward 时序序列)
    ▼
LogAnalyzer.parse_logs()
    │
    ├─► _extract_task_performance()  → fitness
    │
    ├─► 提取 episode_fitnesses 时序
    │
    └─► _evaluate_convergence()  → convergence_status
                                        │
                                        ├── f_mean (增益)
                                        ├── f_std  (收敛)
                                        ├── f_slope (趋势)
                                        └── is_converged (三重交集)
    ▼
metrics 注入 results[i]['metrics']
    │
    ▼
RewardDesignAgent.analyze_results()
    │
    ├─► 优先级1: 完全收敛 → max(fitness)
    ├─► 优先级2: 趋势上升 → max(fitness)
    └─► 优先级3: 盲目Max  → max(fitness)
    │
    ▼
最优代码 + 反思内容
```

---

## 5. 使用说明

### 5.1 真实训练场景

系统会自动完成全链路，无需手动干预：

```python
# 配置好 fitness_config 中的 convergence 参数后
results = simulator.run_parallel(codes, generation)
# results 中每个元素已包含 metrics.convergence_status

# 【重点】必须接收三个返回值，selected_fitness 会传入 memory
best_code, reflection, selected_fitness = agent.analyze_results(results)

# selected_fitness 是降级选拔后的真实适应度
# 会自动传入 memory.save(selected_fitness=selected_fitness)
```

### 5.2 模拟测试场景

`use_real_training=False` 时，`_simulate_training` 会自动注入随机收敛状态，用于快速验证降级选拔逻辑。

### 5.3 调整收敛阈值

如需调整拦截严格程度，修改 `llm_reward_agent/config/llm_config.yaml` 或 `log_analyzer.py` 中对应配置：

```yaml
fitness_config:
  convergence:
    # 建议值 = 总回合数 × 5%~15%，例如 3000 回合建议 alpha=150, beta=300
    alpha: 150   # 增大 → 更重视早期稳定性
    beta: 300    # 增大 → 更重视末期收敛
    v_th: 0.8   # 减小 → 更严格；增大 → 更宽松（多智能体推荐 0.8）
```

> **提示**：对于单智能体环境可将 `alpha/beta` 调回 `20/40`，`v_th` 调至 `0.5` 以获得更严格的质量把控；对于多智能体环境建议保持 `alpha=150, beta=300, v_th=0.8`。

---

## 6. 测试验证

可使用以下方式验证功能是否正常：

```python
from llm_reward_agent.tools.log_analyzer import LogAnalyzer
import numpy as np

analyzer = LogAnalyzer()

# 模拟一个收敛的时序：后期均值升高，方差收缩
converged_seq = list(np.linspace(-1, 2, 30)) + list(np.random.normal(2, 0.2, 40))
result = analyzer._evaluate_convergence(converged_seq)
print(result)
# 期望: is_converged=True, f_mean=True, f_std=True, f_slope=True

# 模拟一个震荡假收敛：均值升高但方差未收缩
noisy_seq = list(np.random.uniform(-1, 3, 30)) + list(np.random.uniform(-1, 3, 40))
result = analyzer._evaluate_convergence(noisy_seq)
print(result)
# 期望: is_converged=False, f_std=False
```

---

## 7. 与其他任务的关联

- **任务一（CoT两阶段管线）**：提供了 MDP 表征先验，使 LLM 生成的奖励函数在结构上更合理，降低被拦截器过滤的概率
- **任务三（EvoLeap变异算子）**：拦截器结果将反馈给 EvoLeap 变异策略，引导生成更有潜力的突变方向
- **任务四（上下文压缩与记忆检索）**：收敛诊断结果作为历史经验的一部分存入记忆，供后续代际参考

---

## 8. 数据流贯通（关键修复）

> ⚠️ **重要说明**：降级选拔算法必须贯穿整个数据流，否则将被架空。

### 8.1 问题背景

在 `analyze_results` 中，我们通过三重拦截器（F_mean, F_std, F_slope）选出了真正收敛的代码（它可能不是适应度绝对值最大的）。但在修复前，`step` 函数、`run_evolution.py` 以及 `memory.py` 中依然在执行野蛮的 `best_fitness = max(...)` 逻辑。

这意味着：系统选拔了正确收敛的代码去繁衍，但在记录历史、绘制曲线和输出最终结果时，依然挂着那个未收敛、靠数据毛刺骗取高分的废弃代码的适应度。

### 8.2 修复方案

为确保数据流贯通，需要修改三个文件：

**① `reward_design_agent.py` — `analyze_results` 方法**

返回值从 `Tuple[str, str]` 扩展为 `Tuple[str, str, float]`：

```python
def analyze_results(self, results: List[Dict]) -> Tuple[str, str, float]:
    # ... 降级选拔逻辑 ...
    return best_result['code'], reflection, best_result['fitness']
```

**② `reward_design_agent.py` — `step` 方法**

接收三个返回值，并显式传递给 memory：

```python
best_code, reflection, best_fitness = self.analyze_results(results)

self.memory.save(
    generation=generation,
    best_code=best_code,
    reflection=reflection,
    all_results=results,
    selected_fitness=best_fitness  # 显式传递过滤后的 fitness
)
```

**③ `memory.py` — `save` 方法**

增加 `selected_fitness` 参数，优先使用传入值而非盲目取 max：

```python
def save(self,
         generation: int,
         best_code: str,
         reflection: str,
         all_results: List[Dict],
         selected_fitness: float = None,  # 【新增参数】
         metadata: Optional[Dict] = None):

    if selected_fitness is not None:
        best_fitness = selected_fitness
    else:
        best_fitness = max(r.get('fitness', -float('inf')) for r in all_results)
```

**④ `run_evolution.py` — `run_initial_generation` 函数**

对齐主控脚本的第零代引导逻辑：

```python
best_code, reflection, best_fitness = agent.analyze_results(results)

agent.memory.save(
    generation=generation,
    best_code=best_code,
    reflection=reflection,
    all_results=results,
    selected_fitness=best_fitness
)
```

### 8.3 修复效果

完成以上修改后，收敛过滤体系才算实现了物理隔离。那些虽然获得了短暂高分，但在随后的回合中因策略崩溃导致方差爆炸的错误代码，将被系统永久抛弃，再也没有机会污染：

- 进化历史曲线
- 反思提示词池
- 最终输出的最优代码

---

## 9. 修订日志

| 版本 | 日期 | 修订内容 |
|------|------|----------|
| v1.1 | 2026-04-09 | 将 `v_th` 从 0.5 调整为 0.8，适应多智能体强化学习环境的非平稳特性，避免过度拦截有潜力的候选。更新文档参数说明与使用建议。 |
| v1.0 | 2026-04-07 | 初始版本，实现三重收敛过滤拦截器（F_mean, F_std, F_slope）与降级选拔算法。 |

