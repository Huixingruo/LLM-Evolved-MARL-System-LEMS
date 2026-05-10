# Bug修复报告

**日期**: 2026-04-08
**问题来源**: 用户运行 `run_evolution.py` 时遇到的运行时问题

---

## Bug #1: OMP Error #15 运行时崩溃

### 现象

```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked
into the program. That is dangerous, since it can degrade performance or cause
incorrect results.
```

### 原因分析

在 Windows 系统上，PyTorch、NumPy 和 MKL（数学核心库）等底层 C++ 库各自链接了不同版本的 OpenMP 运行时。当这些库在同一个进程中被加载时，会导致 `libiomp5md.dll` 和 `libomp.dll` 冲突，造成运行时错误。

### 修复方案

在三个入口文件的**最顶部**（任何其他库加载之前）强制设置环境变量 `KMP_DUPLICATE_LIB_OK='True'`，允许重复加载 OpenMP 库：

#### 1. `run_evolution.py` (主入口)

```python
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

# 强制允许OpenMP库重复加载，解决Windows下OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

#### 2. `MADDPG/main_train.py` (沙盒训练入口)

```python
# 强制允许OpenMP库重复加载，解决Windows下OMP: Error #15
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
...
```

> **注意**：不能在函数或类内部设置此环境变量，必须在模块最顶层、`import torch` 等库之前设置，否则无效。

#### 3. `launcher.py` (并行调度器)

在 `_start_training_process_v2` 和 `_run_single_training_gpu` 两个启动子进程的方法中，通过 `env=os.environ.copy()` 继承主进程的环境变量（包括 `KMP_DUPLICATE_LIB_OK`），因此子进程自动继承，无需额外修改。

### 修复状态

✅ 已修复。

---

## Bug #3: API边界幻觉导致运行时错误 (NameError / AttributeError)

### 现象

```
NameError: name 'CoreEnvLogic' is not defined. Did you mean: 'core_logic'?

AttributeError: 'CustomWorld' object has no attribute 'logic'
```

### 原因分析

在 `prompt_templates.py` 的 `PREDEFINED_ENV_CONTEXT` 中，提供了一个名为 `CoreEnvLogic` 的**文档伪代码类**来帮助 LLM 理解物理常量。然而，LLM 产生了"API边界幻觉（API Boundary Hallucination）"：

1. **NameError**：`CoreEnvLogic` 是代码片段中的示例类，仅供理解物理概念，**绝非**可以在运行时被实例化的真实类。LLM 误以为它是一个真实可导入的类。
2. **AttributeError**：LLM 试图基于自己的直觉去解析 `world` 对象，凭空捏造了 `world.logic`、`world.adversary_params` 等不存在的属性。

在将 LLM 接入代码执行沙盒时，**绝对不能假定 LLM 懂得区分"上下文文档"和"运行时可用API"**。

### 修复方案

在提示词模板中实施**致命红线约束（Anti-Hallucination Guardrails）**：

#### 1. 升级 `cot_analysis_prompt`（添加第5维度：API边界隔离）

在思维链分析中，强制 LLM 先自我声明 API 限制：

```python
## API边界隔离 (API Boundaries)
- **重要**：上方代码片段中的 `CoreEnvLogic` 类仅是**文档伪代码**，用于辅助理解物理概念，**绝非**可以在运行时被实例化的真实类。
- 在真正的 `compute_reward` 函数中，**禁止**调用 `CoreEnvLogic()` 或访问 `world.logic`、`world.adversary_params` 等不存在的属性。
- 如果需要物理常量（如 `size`、`max_speed`、`world_size` 等），必须在函数内部以**局部变量**的形式硬编码声明，例如：`adv_size = 0.075`。
```

#### 2. 升级 `initial_generation_prompt_with_cot`（注入致命红线）

```python
# 致命红线约束 (Anti-Hallucination Guardrails)
- **禁止实例化伪类**：绝对禁止在代码中写出 `CoreEnvLogic()`！
- **禁止虚构属性**：绝对禁止调用 `world.logic`、`world.adversary_params` 等不存在的属性。
- **物理常量硬编码**：如果需要使用物理参数（如智能体的 `size=0.075`、地图大小 `world_size=2.5` 等），必须直接在函数内部以局部变量的形式硬编码声明。
```

#### 3. 升级 `evoleap_prompt`（防止变异时产生幻觉）

同样在进化代的变异提示词中补齐红线约束，防止同样的幻觉在后续代中复发。

### 修复状态

✅ 已修复。

---

## Bug #4: 收敛拦截器被跳过 ("未提取到时序Episode奖励曲线")

### 现象

```
⚠️ 未提取到时序(Episode)奖励曲线，跳过收敛拦截器。
```

### 原因分析

`log_analyzer.py` 中的 `LogAnalyzer.parse_logs()` 方法期望从 `training_log.json` 中读取 `episodes` 数组（每回合的奖励时序），以执行"三重收敛拦截器"（均值增益、方差收敛、斜率趋势校验）。

但 `MADDPG/utils/logger.py` 中的 `TrainingLogger.save_training_log()` **只保存了训练参数和最终统计结果**，并未保存每回合的 `episodes` 时序数组：

```json
// 修复前的 training_log.json 格式
[{
    "训练时间": "...",
    "总回合数": 3000,
    "成功围捕次数": 230,
    "成功围捕率": "7.67%",
    "平均回合步数": "97.7"
    // ❌ 缺少 "episodes" 字段
}]
```

因此 `log_analyzer.py` 的逻辑：

```python
if 'episodes' in log_data:
    episode_fitnesses = [ep.get('total_reward', 0) for ep in log_data['episodes']]
```

永远无法命中，导致收敛拦截器被跳过，所有候选都无法经过三重收敛过滤。

### 修复方案

修改 `MADDPG/utils/logger.py`，在保存训练日志时同步构建 `episodes` 时序数组：

```python
# 构建每回合的时序数据，供log_analyzer.py的收敛拦截器使用
episodes_data = []
for i, reward_sum in enumerate(runner.all_sum_rewards):
    episode_info = {
        "episode": i,
        "total_reward": float(reward_sum)
    }
    if i < len(runner.capture_success_record):
        episode_info["capture_success"] = bool(runner.capture_success_record[i])
    if i < len(runner.episode_steps_record):
        episode_info["steps"] = int(runner.episode_steps_record[i])
    episodes_data.append(episode_info)
```

并将 `episodes_data` 写入 `log_info` 字典的 `"episodes"` 字段。

修复后 `training_log.json` 格式变为：

```json
// 修复后的 training_log.json 格式
[{
    "训练时间": "...",
    "总回合数": 3000,
    "成功围捕次数": 230,
    "成功围捕率": "7.67%",
    "平均回合步数": "97.7",
    "episodes": [
        {"episode": 0, "total_reward": -12.34, "capture_success": false, "steps": 100},
        {"episode": 1, "total_reward": -8.56, "capture_success": false, "steps": 100},
        ...
    ]
}]
```

这样 `log_analyzer.py` 的收敛拦截器就能正确读取并执行三重过滤（均值增益 `F_mean`、方差收敛 `F_std`、趋势判断 `F_slope`），淘汰靠运气拿高分的不稳定代码。

### 修复状态

✅ 已修复。

---

## 修改文件清单

| 文件路径 | 修改内容 |
|---|---|
| `run_evolution.py` | 第20行添加 `os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'` |
| `MADDPG/main_train.py` | 顶部添加 OpenMP 环境变量设置 |
| `MADDPG/utils/logger.py` | `save_training_log` 方法中添加 `episodes` 时序数组构建与写入逻辑 |
| `llm_reward_agent/agent/prompt_templates.py` | 三个提示词方法添加"致命红线约束"：<br>1. `cot_analysis_prompt` 新增第5维度"API边界隔离"<br>2. `initial_generation_prompt_with_cot` 添加禁止伪类实例化和物理常量硬编码约束<br>3. `evoleap_prompt` 添加相同约束防止变异时产生幻觉 |
| `docs/COT_TWO_STAGE_PIPELINE.md` | 更新分析维度从4个到5个，添加API边界隔离说明 |
| `docs/evoLeap_mutate_operator.md` | 添加致命红线约束说明 |
| `docs/BUGFIX_REPORT.md` | 新增 Bug #3: API边界幻觉导致运行时错误的修复记录 |


---

## Bug #5: 奖励分量统计报告未传递给 Phase4 反思阶段（数据管道断点）

### 现象

在 Phase4 反思阶段，发送给 LLM 的 prompt 中缺少奖励分量统计报告（reward_components 和 collaboration_metrics），导致 LLM 仅能基于模糊的 fitness 数值进行诊断，无法进行精准的信用分配分析。

### 原因分析

数据流断点发生在 `log_analyzer.py` 的 `parse_logs()` 方法中。`_find_latest_stats_file()` 方法存在但从未被调用，导致 `reward_component_stats_*.json` 中的 `reward_components` 和 `collaboration_metrics` 无法进入 `metrics` 字典，最终无法传递给 Phase4 反思阶段的 LLM。

### 修复方案

在 `parse_logs()` 中显式调用 `_load_stats_file()`，将统计文件中的数据填充到 `metrics`：

```python
# 读取奖励分量统计文件和协同行为指标
stats_data = self._load_stats_file(sandbox_path)
if stats_data:
    metrics['reward_components'] = stats_data.get('reward_components', {})
    metrics['collaboration_metrics'] = stats_data.get('collaboration_metrics', {})
    print(f"    ✅ 成功读取奖励分量统计 ({len(metrics['reward_components'])} 个分量)")
```

新增 `_load_stats_file()` 方法，封装文件查找和 JSON 解析逻辑。

### 修复状态

✅ 已修复。

### 修改文件清单

| 文件路径 | 修改内容 |
|---|---|
| `llm_reward_agent/tools/log_analyzer.py` | `parse_logs()` 新增调用 `_load_stats_file()`；新增 `_load_stats_file()` 方法 |
| `docs/REFACTORED_MODULES.md` | 更新 `LogAnalyzer.parse_logs()` API 文档；新增 `_load_stats_file()` 和 `_find_latest_stats_file()` API 文档 |
| `docs/BUGFIX_REPORT.md` | 新增 Bug #5 修复记录 |

