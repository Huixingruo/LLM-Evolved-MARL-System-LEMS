# LEMS 项目模块重构说明

> 本文档记录了 LEMS 项目在重构过程中所做的代码清理工作，说明了重构后的代码架构、主要文件职责以及关键 API，供其他开发者（或未来的自己）参考。

---

## 1. 模块概述

LEMS（LLM-driven Evolution of Multi-Agent Reward System）将大语言模型（LLM）与多智能体强化学习（MARL）结合，实现奖励函数的自动设计与优化。

系统由两个核心子系统组成：

```
LEMS
├── MADDPG/              # 多智能体深度确定性策略梯度训练框架
│   ├── envs/            # PettingZoo 环境包装、奖励函数、可插拔架构
│   ├── agents/          # DDPG / MADDPG 智能体实现
│   └── utils/           # 训练运行器、日志、可视化工具
│
├── llm_reward_agent/    # LLM 驱动的奖励函数进化系统
│   ├── agent/           # 核心 Agent（LLM接口、提示词模板、记忆管理）
│   └── tools/           # 仿真工具（沙盒管理、日志分析、上下文提取）
│
├── launcher.py          # 并行训练调度器（CPU / GPU）
├── run_evolution.py     # 进化主流程入口
└── visualization/      # 进化过程可视化
```

### 自动设计奖励函数的工作流

```
┌─────────────────────────────────────────────────────────────┐
│                    单代进化循环 (agent.step)                   │
│                                                             │
│  1. generate_candidates()                                   │
│     LLM 基于当前代上下文生成 N 个候选 reward_function.py          │
│                                                             │
│  2. SimulationTool.run_parallel()                            │
│     ├─ SandboxManager.create_sandboxes()                     │
│     │   每个候选 → 独立沙盒目录（含完整 MADDPG 代码副本）         │
│     │                                                        │
│     └─ ParallelLauncher.run_parallel()                      │
│         CPU: multiprocessing.Pool                            │
│         GPU: subprocess 多进程 + CUDA_VISIBLE_DEVICES         │
│         每个进程运行 MADDPG/main_train.py                      │
│                                                             │
│  3. LogAnalyzer.parse_logs()                                │
│     解析每个沙盒的 training_log.json → 提取指标                 │
│     计算 Fitness = w1·成功率 - w2·捕获时间 + w3·协作分          │
│                                                             │
│  4. analyze_results()                                        │
│     LLM 分析训练结果 → 选择最优候选 + 生成反思                  │
│     反思内容存入 EvolutionaryMemory                          │
│                                                             │
│  5. EvolutionaryMemory.save()                               │
│     本代最优代码、反思、所有候选结果 → JSON 文件持久化           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心文件说明

### 2.1 MADDPG 训练子系统

| 文件 | 职责 |
|------|------|
| `MADDPG/envs/simple_tag_env.py` | PettingZoo 环境包装器（`Custom_raw_env`）。整合物理世界、自定义动力学、可插拔奖励函数和 matplotlib 渲染。追捕者奖励调用 `reward_function.compute_reward()`，逃跑者奖励在环境中固定实现。 |
| `MADDPG/envs/reward_function.py` | **可插拔模块**——LLM 的设计目标。仅包含追捕者奖励函数 `compute_reward(agent_name, observation, global_state, actions, world)`。返回 `(scalar_reward, components_dict)`。 |
| `MADDPG/envs/custom_agents_dynamics.py` | 自定义物理世界 `CustomWorld`，重写 `integrate_state()` 实现自定义阻尼、接触力和速度限制。 |
| `MADDPG/agents/maddpg/MADDPG_agent.py` | MADDPG 算法实现。管理多个 DDPG 智能体的 replay buffer，centralized critic，全局 action/observation 选择。 |
| `MADDPG/agents/maddpg/DDPG_agent.py` | 单 DDPG 智能体实现。包含 `MLPNetworkActor`、`MLPNetworkCritic`、目标网络软更新、梯度裁剪。 |
| `MADDPG/utils/runner.py` | 训练循环编排器。管理 agent↔environment 交互、experience 存储、模型更新，早停机制。集成 `RewardComponentLogger` 记录奖励分量。 |
| `MADDPG/main_train.py` | 单次 MADDPG 训练入口脚本。解析命令行参数，创建环境/Agent，驱动训练循环，保存模型和日志。 |
| `MADDPG/main_evaluate.py` | 基础评估脚本。加载已训练模型，调用 `runner.evaluate()` 输出成功率等指标。 |
| `MADDPG/main_evaluate_matplotlib.py` | 完整渲染评估脚本。支持实时动画、PNG 序列、GIF 生成四种显示模式。 |
| `MADDPG/utils/reward_logger.py` | `RewardComponentLogger`——记录每个 reward components（`distance_reward`、`collision_penalty` 等）和协作指标（`encirclement_angle_std`、`formation_quality`）。 |
| `MADDPG/utils/logger.py` | `TrainingLogger`——保存训练元数据、reward 曲线到 `.pkl` 文件。 |

### 2.2 LLM Reward Agent 子系统

| 文件 | 职责 |
|------|------|
| `llm_reward_agent/agent/reward_design_agent.py` | **核心编排器** `RewardDesignAgent`。初始化 LLM/Memory/PromptBuilder，驱动 `step()` 进化循环（生成→训练→分析→记忆）。 |
| `llm_reward_agent/agent/llm_interface.py` | **LLM 统一接口** `LLMInterface`。支持 OpenAI/DeepSeek，封装 API key 管理、超时、多并发请求和重试逻辑。提供 `generate()` 和 `analyze()` 两个 API。 |
| `llm_reward_agent/agent/prompt_templates.py` | **提示词模板库** `PromptTemplates`。包含 `SYSTEM_MESSAGE`、`initial_generation_prompt_with_predefined_context`、`evolution_prompt_with_predefined_context`、`reflection_prompt` 四个静态方法，以及预定义环境上下文 `PREDEFINED_ENV_CONTEXT`。 |
| `llm_reward_agent/agent/memory.py` | **进化记忆库** `EvolutionaryMemory`。保存/加载进化历史（JSON 格式），计算历史最优 `get_best_ever()`，委托绘图给 `visualization.evolution_plot.EvolutionPlotter`。 |
| `llm_reward_agent/tools/sandbox_manager.py` | **沙盒管理器** `SandboxManager`。为每个候选奖励函数创建独立训练目录（复制 MADDPG 代码，写入 `reward_function.py`），支持清理。 |
| `llm_reward_agent/tools/simulation_tool.py` | **仿真工具** `SimulationTool`。整合 `SandboxManager` 和 `ParallelLauncher`，提供统一的 `run_parallel(codes, generation)` 接口。 |
| `llm_reward_agent/tools/log_analyzer.py` | **日志分析器** `LogAnalyzer`。解析 `training_log.json`，计算综合 `fitness`，提取 reward components 和协作指标。可生成人类可读的分析报告。 |
| `llm_reward_agent/tools/context_extractor.py` | **环境上下文提取器** `LLMFriendlyContextExtractor` + 便捷入口 `EnvironmentContextExtractor`。通过 AST 解析和运行时反射提取 LLM 友好的环境描述。当前 Agent 使用预定义上下文，此类仅用于演示/测试。 |

### 2.3 入口与工具

| 文件 | 职责 |
|------|------|
| `launcher.py` | **并行训练调度器** `ParallelLauncher`。CPU 模式（`multiprocessing.Pool`）和 GPU 模式（`subprocess.Popen` + `CUDA_VISIBLE_DEVICES`）。还提供 `run_sequential()` 和 `run_gpu_sequential()` 串行执行方法（用于调试）。 |
| `run_evolution.py` | **进化主流程入口**。整合 Agent 初始化、进化循环、结果摘要保存和最终代码导出。支持 `--initial_code` 从人工设计代码开始（代0），也支持纯 Zero-Shot 进化。 |
| `visualization/evolution_plot.py` | **进化可视化工具** `EvolutionPlotter`。从 `evolution_archive/` 读取数据，绘制进化曲线、fitness 分布箱线图、成功率和综合仪表盘。 |

---

## 3. 关键类与函数 API

### 3.1 `RewardDesignAgent` — 奖励函数进化 Agent

位于 `llm_reward_agent/agent/reward_design_agent.py`

```python
class RewardDesignAgent:
    def __init__(self, config_path: str = "llm_reward_agent/config/llm_config.yaml")
        """
        初始化 Agent。
        - 加载 YAML 配置（LLM 参数、训练参数、日志路径）
        - 创建 LLMInterface（支持 OpenAI / DeepSeek）
        - 创建 EvolutionaryMemory（自动创建存档目录）
        - 创建 PromptTemplates
        """
```

```python
    def initialize(self, env_file_path: str = None, task_description: str = None) -> None
        """
        初始化任务环境上下文。
        使用预定义的 PREDEFINED_ENV_CONTEXT，无需动态提取。
        如果传入 task_description 则覆盖默认值。

        Args:
            env_file_path: 环境文件路径（可选，已使用预定义上下文）
            task_description: 自定义任务描述（可选）
        """
```

```python
    def step(self, generation: int, use_real_training: bool = True) -> dict
        """
        执行单代进化（生成→训练→分析→记忆）。

        Args:
            generation: 当前代数
            use_real_training: True=真实 MADDPG 训练，False=模拟训练

        Returns:
            dict: {
                'generation': int,
                'best_code': str,
                'best_fitness': float,
                'reflection': str,
                'all_results': List[dict]
            }
        """
```

```python
    def generate_candidates(self, generation: int) -> List[str]
        """
        调用 LLM 生成候选奖励函数代码。
        内部处理解析、语法检查、失败重试。

        Returns:
            List[str]: N 个候选 reward_function.py 代码字符串
        """
```

```python
    def analyze_results(self, results: List[dict]) -> Tuple[str, str, float]
        """
        调用 LLM 分析训练结果，选择最优候选并生成反思。
        实装降级选拔算法，根据三重收敛拦截器结果选择最优候选。
        包含 3 次重试机制，全部失败则保存提示词并抛出异常。

        Args:
            results: SimulationTool.run_parallel() 返回的训练结果列表

        Returns:
            Tuple[str, str, float]: (best_code, reflection, selected_fitness)
                - best_code: 降级选拔算法选定的最优代码
                - reflection: LLM 生成的反思内容
                - selected_fitness: 降级选拔算法选定的真实适应度（过滤后）
        """
```

### 3.2 `LLMInterface` — LLM 统一接口

位于 `llm_reward_agent/agent/llm_interface.py`

```python
class LLMInterface:
    def __init__(self,
                 provider: str = "openai",    # "openai" | "deepseek"
                 model_name: str = "gpt-5.1",
                 api_key: str = None,          # 优先从环境变量读取
                 base_url: str = "https://api.vectorengine.ai/v1",
                 timeout: int = 120,
                 max_retries: int = 3)
```

```python
    def generate(self,
                prompt: str,
                n: int = 1,                   # 并发生成数量
                temperature: float = 0.7,
                max_tokens: int = 2000,
                system_message: str = None) -> List[str]
        """
        生成 N 个不同回复（通过并发多线程模拟 n>1）。
        内部强制 n=1，OpenAI API 不直接支持多候选时自动并发。

        Returns:
            List[str]: N 个模型回复字符串
        """
```

```python
    def analyze(self, prompt: str, ...) -> str
        """
        单次分析调用（生成反思/评价）。内部调用 generate(n=1)。
        如果 generate 返回空列表则抛出 RuntimeError。

        Returns:
            str: LLM 分析结果字符串
        """
```

### 3.3 `EvolutionaryMemory` — 进化记忆

位于 `llm_reward_agent/agent/memory.py`

```python
class EvolutionaryMemory:
    def __init__(self, save_dir: str = "experiments/evolution_archive")
        # 创建目录，初始化空 history
```

```python
    def save(self,
             generation: int,
             best_code: str,
             reflection: str,
             all_results: List[dict],
             selected_fitness: float = None,  # 【新增】降级选拔后的真实适应度
             metadata: dict = None) -> None
        """
        保存一代记录。
        优先使用传入的 selected_fitness（降级选拔后的真实适应度），
        而不是盲目取 max()，确保进化历史曲线反映真实收敛质量。

        Args:
            generation: 代数
            best_code: 本代最优代码
            reflection: 反思内容
            all_results: 所有候选的结果列表
            selected_fitness: 【新增】降级选拔算法选定的真实适应度（优先使用）
            metadata: 额外的元数据

        - 添加到 self.history
        - 更新 self.metadata
        - 写入 generation_{g:03d}.json
        - 写入 metadata.json
        """
```

```python
    def load_history(self) -> List[dict]
        """
        从 save_dir 加载所有 generation_*.json。
        启动时自动调用以恢复历史。
        """
```

```python
    def get_best_ever(self) -> dict
        """
        获取历史最优记录（metadata['best_generation'] 指向的记录）。

        Returns:
            dict: {generation, best_code, reflection, best_fitness, ...}
        """
```

```python
    def get_fitness_history(self) -> List[float]
        # 返回每代的 best_fitness 列表
```

```python
    def plot_evolution_curve(self, save_path: str = None) -> None
        """
        委托给 visualization.evolution_plot.EvolutionPlotter 绘图。
        如果绘图失败（ImportError 或异常）静默跳过。
        """
```

### 3.4 `LogAnalyzer` — 日志分析器

位于 `llm_reward_agent/tools/log_analyzer.py`

```python
class LogAnalyzer:
    def __init__(self, fitness_config: dict = None)
        # fitness_config 默认权重:
        #   success_rate: 1.0,  capture_time: -0.001,
        #   formation_quality: 0.3,  collision_penalty: -0.5
```

```python
    def parse_logs(self, sandbox_path: str) -> dict
        """
        解析沙盒目录中的训练日志，提取完整性能指标。

        数据来源（按优先级）：
        1. training_log.json → 提取任务性能（成功率、平均捕获时间）
        2. reward_component_stats_*.json → 提取奖励分量统计和协同行为指标
        3. episodes 时序数组 → 执行三重收敛拦截器

        【关键断点修复】：parse_logs() 内部会调用 _load_stats_file() 加载最新的
        reward_component_stats_*.json，将 reward_components 和 collaboration_metrics
        填充进 metrics 字典。这两个字段是 _format_logs() 和 reflection_prompt
        的数据来源——若此处为空，Phase4 反思阶段 LLM 将无法获得奖励分量统计报告。

        Args:
            sandbox_path: 沙盒目录路径

        Returns:
            dict: {
                'fitness': float,
                'success_rate': float,
                'avg_capture_time': float,
                'reward_components': dict,      # 新增：从 *.json 加载
                'collaboration_metrics': dict,  # 新增：从 *.json 加载
                'raw_data': dict,
                'convergence_status': dict
            }
        """
解析沙盒目录中的 training_log.json。

        Args:
            sandbox_path: 沙盒目录路径

        Returns:
            dict: {
                'fitness': float,
                'success_rate': float,
                'avg_capture_time': float,
                'reward_components': dict,
                'collaboration_metrics': dict,
                'raw_data': dict
            }
        """
```

```python
    def _load_stats_file(self, sandbox_path: str) -> dict
        """
        加载最新的 reward_component_stats_*.json 文件。

        内部调用 _find_latest_stats_file() 查找沙盒目录下
        MADDPG/logs/reward_component_stats_*.json，返回其中包含的
        reward_components 和 collaboration_metrics 两个顶级字段。

        Returns:
            dict: JSON 文件的完整内容（找不到则返回空字典）
        """
```

```python
    def _find_latest_stats_file(self, sandbox_path: str) -> Optional[str]
        """
        查找沙盒目录下最新的 reward_component_stats_*.json 文件路径。
        """
```

```python
    def calculate_fitness(self, metrics: dict) -> float
        """
        根据配置权重计算综合 fitness。

        公式: fitness = w1·成功率 + w2·(-avg_capture_time/max_time)
                    + w3·formation_quality + w4·collision_penalty
        """
```

### 3.5 `SimulationTool` — 仿真执行工具

位于 `llm_reward_agent/tools/simulation_tool.py`

```python
class SimulationTool:
    def __init__(self,
                 base_dir: str = "experiments",
                 max_workers: int = 4,
                 timeout: int = 1200,
                 episode_num: int = 100,
                 use_gpu: bool = True)
        # 初始化 SandboxManager + ParallelLauncher
```

```python
    def run_parallel(self, codes: List[str], generation: int) -> List[dict]
        """
        并行执行多个候选代码的训练。

        流程:
        1. SandboxManager.create_sandboxes() → 每个代码写入独立目录
        2. ParallelLauncher.run_parallel() → 并行启动训练进程
        3. 收集每个沙盒的 LogAnalyzer.parse_logs() 结果

        Returns:
            List[dict]: 每个候选的 {id, status, fitness, metrics, ...}
        """
```

### 3.6 `SandboxManager` — 沙盒管理器

位于 `llm_reward_agent/tools/sandbox_manager.py`

```python
class SandboxManager:
    def __init__(self, base_dir: str = "experiments")
        # 创建 base_dir
```

```python
    def create_sandboxes(self, generation: int, codes: List[str]) -> List[str]
        """
        为每个候选创建独立沙盒目录。

        每个沙盒包含:
        - MADDPG/ 的完整代码副本（deepcopy，排除 __pycache__）
        - MADDPG/envs/reward_function.py（由 LLM 生成）

        Returns:
            List[str]: 沙盒目录绝对路径列表
        """
```

### 3.7 `ParallelLauncher` — 并行训练调度器

位于 `launcher.py`

```python
class ParallelLauncher:
    def __init__(self,
                 max_workers: int = 4,
                 timeout: int = 12000,
                 episode_num: int = 100,
                 use_gpu: bool = True,
                 gpu_ids: List[int] = None)
        """
        GPU 模式自动检测 CUDA 可用性，不可用则回退到 CPU。
        """
```

```python
    def run_parallel(self, sandbox_paths: List[str]) -> List[dict]
        """
        根据 use_gpu 选择 CPU 或 GPU 并行模式。

        CPU: multiprocessing.Pool.map(_run_single_training, paths)
        GPU: subprocess.Popen → CUDA_VISIBLE_DEVICES → 循环分配 GPU

        Returns:
            List[dict]: 每个沙盒的训练结果
        """
```

```python
    def run_sequential(self, sandbox_paths: List[str]) -> List[dict]
    def run_gpu_sequential(self, sandbox_paths: List[str]) -> List[dict]
        # 串行执行方法（用于调试）
```

### 3.8 可插拔奖励函数接口

位于 `MADDPG/envs/reward_function.py`

```python
def compute_reward(agent_name: str,
                   observation: np.ndarray,
                   global_state: dict,
                   actions: dict,
                   world: World) -> Tuple[float, dict]:
    """
    可插拔奖励函数接口。LLM 生成此函数的实现。

    global_state 包含:
        agent_positions: np.ndarray (n_agents, 2)
        agent_velocities: np.ndarray (n_agents, 2)
        prey_position: np.ndarray (2,)
        prey_velocity: np.ndarray (2,)
        distances_to_prey: np.ndarray (n_adversaries,)
        inter_agent_distances: np.ndarray (n_agents, n_agents)
        is_adversary: bool          # 当前是否为追捕者
        adversary_indices: list      # 所有追捕者索引
        prey_indices: list           # 所有猎物索引
        world_size: float
        capture_threshold: float

    Returns:
        float: 总奖励值
        dict:  奖励分量（用于日志分析），如 {
                   'distance_reward': float,
                   'collision_penalty': float,
                   'formation_reward': float,
                   'boundary_penalty': float
               }
    """
```

---

## 4. 重构说明

本次重构**严格遵守不改变任何现有功能和算法逻辑**的原则，仅消除以下冗余。

### 4.1 清理的冗余文件

| 被删除文件 | 原因 |
|-----------|------|
| `MADDPG/main_evaluate_debug.py` | 与 `main_evaluate.py` 重复，仅多每步打印；`main_evaluate_matplotlib.py` 已提供完整调试功能 |
| `MADDPG/main_evaluate_save_render2gif.py` | PettingZoo 原生渲染方案，与 `main_evaluate_matplotlib.py` 重复 |
| `run_evolution_with_code.py` | 与 `run_evolution.py` 重复；合并为单一入口，`--initial_code` 参数支持从指定代码开始 |
| `docs/PHASE*.md`、`docs/MODIFICATION_GUIDE.md`、`docs/2026年2月12日_改动.md` 等 12 个阶段文档 | 功能已被 `README.md` 和 `USER_MANUAL.md` 覆盖，内容已过时 |

### 4.2 清理的死代码

| 文件 | 清理内容 |
|------|---------|
| `launcher.py` | 删除了第 274-356 行三重引号包裹的废弃 `_wait_training_process` 方法（~83 行），删除了未使用的 `import json` |
| `llm_reward_agent/agent/prompt_templates.py` | 删除了未使用的 `initial_generation_prompt()` 和 `evolution_prompt()` 两个旧版本方法（~150 行），删除了未调用的 `code_fix_prompt()` 方法 |
| `llm_reward_agent/agent/reward_design_agent.py` | 删除了已废弃的 `_parse_variants()` 方法（约 24 行） |
| `llm_reward_agent/agent/memory.py` | 将 `plot_evolution_curve()` 的独立 matplotlib 实现（约 58 行）替换为对 `EvolutionPlotter` 的委托调用，消除绘图代码重复 |
| `llm_reward_agent/tools/context_extractor.py` | 将 `EnvironmentContextExtractor` 子类中的 `extract_skeleton()`、`format_for_llm()`、`estimate_token_count()` 三个包装方法提升到父类 `LLMFriendlyContextExtractor`，子类仅保留初始化路径查找逻辑 |

### 4.3 代码行数变化（主要文件）

| 文件 | 重构前约 | 重构后约 | 变化 |
|------|---------|---------|------|
| `launcher.py` | 532 行 | 448 行 | -84 行 |
| `llm_reward_agent/agent/prompt_templates.py` | ~700 行 | ~480 行 | -~220 行 |
| `llm_reward_agent/agent/reward_design_agent.py` | ~580 行 | ~556 行 | -24 行 |
| `llm_reward_agent/agent/memory.py` | ~315 行 | ~270 行 | -45 行 |
| `llm_reward_agent/tools/context_extractor.py` | ~500 行 | ~490 行 | -10 行（合并方法） |

**合计精简约 383 行冗余代码**，同时删除了 4 个重复脚本和 12 个过时文档。

---

## 5. 快速使用

```bash
# 依赖安装
pip install -r MADDPG/utils/pip-requirements.txt

# 配置 LLM（复制并修改配置文件）
cp llm_reward_agent/config/llm_config.yaml.example llm_reward_agent/config/llm_config.yaml
# 编辑 llm_reward_agent/config/llm_config.yaml 填入 API key

# 快速模拟测试（无需真实训练，3 代）
python run_evolution.py --num_generations 3 --no-real-training

# 真实进化（5 代，每代 3000 回合，4 并行）
python run_evolution.py --num_generations 5 --episode_num 3000 --max_workers 4

# 从人工设计代码开始进化（代 0 使用指定代码）
python run_evolution.py --initial_code MADDPG/envs/reward_function.py --num_generations 5

# 评估已训练模型
cd MADDPG && python main_evaluate.py
# 或使用 matplotlib 渲染
cd MADDPG && python main_evaluate_matplotlib.py
```
