# LEMS 详细实现计划
# LLM驱动的多智能体强化学习奖励函数自动生成系统 - 完整开发蓝图

> **文档版本**: v1.0  
> **创建日期**: 2026-02-02  
> **预计总工期**: 7-11周（约2-3个月）

---

## 📑 目录

1. [项目概述](#1-项目概述)
2. [系统架构设计](#2-系统架构设计)
3. [五阶段实施计划](#3-五阶段实施计划)
4. [关键技术实现细节](#4-关键技术实现细节)
5. [风险评估与应对](#5-风险评估与应对)
6. [质量保障与测试](#6-质量保障与测试)
7. [论文撰写指南](#7-论文撰写指南)
8. [附录：核心代码模板](#8-附录核心代码模板)

---

## 1. 项目概述

### 1.1 研究动机

**现状问题**：
- 多智能体强化学习中，奖励函数设计高度依赖人类专家经验
- 需要反复调试权重系数（$\alpha, \beta, \gamma$...）
- 对于复杂协同任务（如围捕），人工设计往往次优且耗时

**解决方案**：
- 引入大语言模型（LLM）作为"自动化奖励工程师"
- 通过进化算法（Evolutionary Search）迭代优化奖励函数代码
- 基于训练反馈进行反思（Reflection）指导下一轮改进

### 1.2 核心创新点

| 创新维度 | 具体内容 | 学术价值 |
|---------|---------|---------|
| **方法创新** | LLM生成Python代码而非符号表达式 | 首次将EUREKA应用于MARL |
| **任务特性** | 针对多智能体协同设计专门指标 | 填补围捕任务奖励设计空白 |
| **工程实现** | 沙盒并行验证 + 详细日志反馈 | 提供可复现的开源框架 |

### 1.3 预期成果

1. **技术成果**：
   - 完整的LLM-MARL集成系统
   - 可复用的奖励函数生成框架
   - 开源代码与详细文档

2. **实验结果**：
   - LLM设计的奖励函数 vs 人工设计的性能对比
   - 进化代数与性能提升的关系分析
   - 不同LLM（GPT-4, Claude等）的对比实验

3. **学术产出**：
   - 毕业论文（中文/英文）
   - 会议论文投稿（ICRA/IROS/NeurIPS Workshop）
   - 技术博客与开源项目

---

## 2. 系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户输入                                    │
│  • 环境代码（simple_tag_env.py）                              │
│  • 任务描述（"3个智能体协同围捕1个目标，避免碰撞"）              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         【Agent Core】RewardDesignAgent                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Brain (LLM)                                         │   │
│  │  • 读取环境代码，理解物理约束                          │   │
│  │  • 生成K个候选奖励函数代码                             │   │
│  │  • 分析训练日志，生成Reflection                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tools                                               │   │
│  │  • CodeWriter: 写入.py文件                            │   │
│  │  • SimulationTool: 调用launcher.py并行训练             │   │
│  │  • LogAnalyzer: 解析TensorBoard日志                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Memory (EvolutionaryArchive)                        │   │
│  │  • 存储历史最优代码                                    │   │
│  │  • 记录每代的Reflection                               │   │
│  │  • 生成进化树可视化                                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         【Execution Layer】并行训练沙盒                       │
│                                                              │
│  experiments/                                               │
│  ├── candidate_0/  ← GPU0 / CPU核心0                        │
│  │   ├── reward_function.py                                │
│  │   ├── train.py (软链接)                                  │
│  │   └── logs/                                             │
│  ├── candidate_1/  ← GPU0 / CPU核心1                        │
│  ├── candidate_2/  ← GPU1 / CPU核心2                        │
│  └── candidate_3/  ← GPU1 / CPU核心3                        │
│                                                              │
│  launcher.py (Dispatcher)                                   │
│  • Python multiprocessing.Pool                             │
│  • 资源调度与超时控制                                        │
│  • 结果收集与错误处理                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         【Feedback Loop】反馈分析                            │
│                                                              │
│  • 性能指标：success_rate, capture_time, safety_score       │
│  • 奖励分量：dist_reward_mean, collision_penalty_mean       │
│  • 协同指标：encirclement_angle_std, min_agent_distance     │
│                                                              │
│  → 生成自然语言报告 → 反馈给LLM → 下一代进化                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流设计

**输入数据**：
```python
{
    "env_code": "完整的环境源代码字符串",
    "task_description": "任务自然语言描述",
    "parent_code": "上一代最优奖励函数代码（第2代起）",
    "reflection": "上一代的训练反思（第2代起）"
}
```

**中间数据**：
```python
{
    "candidate_codes": [
        {
            "id": 0,
            "code": "def compute_reward(...):\n    ...",
            "syntax_valid": True
        },
        ...
    ]
}
```

**输出数据**：
```python
{
    "best_code": "本代最优代码",
    "fitness": 0.85,
    "metrics": {
        "success_rate": 0.85,
        "avg_capture_time": 45.2,
        "reward_components": {
            "dist_reward": 1.23,
            "collision_penalty": -0.15,
            "formation_reward": 0.67
        }
    },
    "reflection": "本代改进总结与下代建议"
}
```

### 2.3 文件目录结构（最终版）

```
LEMS/
├── MADDPG/                              # 现有基础（已完成）
│   ├── agents/
│   ├── envs/
│   │   ├── simple_tag_env.py
│   │   └── custom_agents_dynamics.py
│   ├── utils/
│   │   ├── runner.py
│   │   └── logger.py
│   ├── main_train.py
│   └── models/
│
├── llm_reward_agent/                    # 【核心新增】
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── reward_design_agent.py       # 主Agent类
│   │   ├── prompt_templates.py          # LLM提示词库
│   │   └── memory.py                    # 进化记忆管理
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── code_writer.py               # 文件写入工具
│   │   ├── simulation_tool.py           # 并行训练调度
│   │   └── log_analyzer.py              # 日志解析与指标提取
│   ├── workflow/
│   │   ├── __init__.py
│   │   └── evolution_graph.py           # LangGraph流程定义
│   └── config/
│       ├── llm_config.yaml              # LLM配置（API Key等）
│       └── evolution_config.yaml        # 进化参数配置
│
├── reward_templates/                    # 【新增】奖励模板库
│   ├── __init__.py
│   ├── base_reward.py                   # 基础模板
│   └── components.py                    # 可组合组件
│
├── experiments/                         # 【运行时生成】
│   ├── generation_0/
│   │   ├── candidate_0/
│   │   ├── candidate_1/
│   │   ├── ...
│   │   └── summary.json
│   ├── generation_1/
│   └── ...
│
├── launcher.py                          # 【新增】并行训练主控
├── run_evolution.py                     # 【新增】进化主程序入口
├── README.md
├── IMPLEMENTATION_PLAN.md               # 本文件
└── requirements_llm.txt                 # LLM相关依赖
```

---

## 3. 五阶段实施计划

### 【阶段一】环境接口标准化（1-2周）

#### 目标
让LLM能够"读懂"环境，让生成的代码能够"插入"训练流程。

#### 任务清单

##### 任务1.1：奖励逻辑解耦（关键）

**当前问题**：
- 奖励计算逻辑散布在 `simple_tag_env.py` 的 `reward()` 方法中
- LLM生成的代码无法直接替换

**改进方案**：
```python
# 【新增文件】 MADDPG/envs/reward_function.py

def compute_reward(agent_name, observation, global_state, actions, world):
    """
    可插拔的奖励函数接口
    
    Args:
        agent_name (str): 当前智能体名称 (e.g., "adversary_0")
        observation (np.ndarray): 当前智能体的观测向量
        global_state (dict): 全局状态信息
            {
                'agent_positions': np.ndarray,  # shape: (n_agents, 2)
                'prey_position': np.ndarray,     # shape: (2,)
                'distances_to_prey': np.ndarray, # shape: (n_agents,)
                'inter_agent_distances': np.ndarray  # shape: (n_agents, n_agents)
            }
        actions (dict): 所有智能体的动作 {agent_name: action_vector}
        world (World): PettingZoo的World对象（用于读取物理参数）
    
    Returns:
        reward (float): 标量奖励值
        components (dict): 奖励分量字典（用于日志分析）
            {
                'distance_reward': float,
                'collision_penalty': float,
                'formation_reward': float,
                'energy_cost': float
            }
    """
    # LLM将生成这里的代码！
    # 默认实现（人工设计的基准版本）
    components = {}
    
    # 1. 距离奖励：鼓励接近猎物
    dist_to_prey = global_state['distances_to_prey'][int(agent_name.split('_')[1])]
    components['distance_reward'] = -0.1 * dist_to_prey
    
    # 2. 碰撞惩罚：避免与队友碰撞
    min_dist_to_teammates = np.min(global_state['inter_agent_distances'][...])
    if min_dist_to_teammates < 0.2:
        components['collision_penalty'] = -10.0
    else:
        components['collision_penalty'] = 0.0
    
    # 3. 队形奖励：保持包围圈均匀
    # （这里简化，完整版需要计算角度方差）
    components['formation_reward'] = 0.0
    
    # 4. 能耗惩罚
    action = actions[agent_name]
    components['energy_cost'] = -0.01 * np.sum(action ** 2)
    
    total_reward = sum(components.values())
    return total_reward, components
```

**修改 `simple_tag_env.py`**：
```python
# 在文件顶部导入
from . import reward_function

# 在 reward() 方法中调用
def reward(self, agent):
    # ... 原有代码准备 global_state ...
    total_rew, components = reward_function.compute_reward(
        agent.name, 
        self.observe(agent), 
        global_state, 
        self.current_actions,
        self.world
    )
    
    # 保存分量用于日志
    self.last_reward_components[agent.name] = components
    
    return total_rew
```

**交付物**：
- ✅ `reward_function.py` 独立模块
- ✅ 修改后的 `simple_tag_env.py`
- ✅ 单元测试：验证奖励计算正确性

---

##### 任务1.2：增强日志系统

**当前问题**：
- 只记录总奖励（total_reward），LLM无法知道哪个组件工作、哪个失效

**改进方案**：

**修改 `MADDPG/utils/runner.py`**：
```python
class RUNNER:
    def __init__(self, ...):
        # 新增：奖励分量统计器
        self.reward_component_stats = {
            agent_name: {
                'distance_reward': [],
                'collision_penalty': [],
                'formation_reward': [],
                'energy_cost': []
            } for agent_name in self.env.agents
        }
        
        # 新增：协同行为指标
        self.collaboration_metrics = {
            'encirclement_angle_std': [],
            'min_agent_distance': [],
            'capture_success': []
        }
    
    def step(self, ...):
        # 在每个step后收集
        for agent_name in self.env.agents:
            components = self.env.last_reward_components.get(agent_name, {})
            for key, value in components.items():
                self.reward_component_stats[agent_name][key].append(value)
        
        # 计算协同指标
        self.collaboration_metrics['encirclement_angle_std'].append(
            self._compute_encirclement_angle_std()
        )
        ...
    
    def _compute_encirclement_angle_std(self):
        """计算围捕角度的标准差（越小越均匀）"""
        adversary_positions = ...
        prey_position = ...
        angles = np.arctan2(
            adversary_positions[:, 1] - prey_position[1],
            adversary_positions[:, 0] - prey_position[0]
        )
        # 理想情况：3个智能体应该相隔120度
        ideal_separation = 2 * np.pi / 3
        angle_diffs = np.diff(np.sort(angles))
        return np.std(angle_diffs - ideal_separation)
```

**新增 `MADDPG/utils/reward_logger.py`**：
```python
class RewardComponentLogger:
    """专门记录奖励分量的日志器"""
    
    def save_statistics(self, stats, filepath):
        """
        保存为JSON，方便LLM读取
        
        输出格式：
        {
            "reward_components": {
                "distance_reward_mean": 1.23,
                "distance_reward_std": 0.45,
                "collision_penalty_mean": -0.15,
                ...
            },
            "collaboration_metrics": {
                "encirclement_angle_std_mean": 0.23,
                ...
            },
            "task_performance": {
                "success_rate": 0.75,
                "avg_capture_time": 48.5
            }
        }
        """
        ...
```

**交付物**：
- ✅ 修改后的 `runner.py`
- ✅ 新增 `reward_logger.py`
- ✅ 测试：运行一次训练，验证日志正确生成

---

##### 任务1.3：上下文提取脚本

**目标**：
自动提取环境代码的"关键信息"，减少LLM的Token消耗。

**新增 `llm_reward_agent/tools/context_extractor.py`**：
```python
import ast
import inspect

class EnvironmentContextExtractor:
    """提取环境代码的核心信息"""
    
    def extract_skeleton(self, env_file_path):
        """
        提取环境代码骨架
        
        返回：
        {
            "observation_space": "Box(16,)",
            "action_space": "Box(2,)",
            "key_methods": ["reset", "step", "reward"],
            "physical_constants": {
                "max_force": 1.0,
                "capture_threshold": 0.5
            },
            "code_snippet": "关键代码片段（不超过500行）"
        }
        """
        with open(env_file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # 1. 找到环境类定义
        env_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and 'Env' in node.name:
                env_class = node
                break
        
        # 2. 提取观测和动作空间（通过正则或AST）
        ...
        
        # 3. 提取物理常量（如max_force）
        ...
        
        return context_dict
```

**交付物**：
- ✅ `context_extractor.py`
- ✅ 测试：对 `simple_tag_env.py` 提取，验证输出正确

---

#### 阶段一验收标准

- [x] 奖励函数可以独立替换（手动替换一个不同的 `reward_function.py` 并成功运行）
- [x] 训练日志包含奖励分量统计（JSON格式）
- [x] 环境上下文可以自动提取（生成<1000 Token的精简描述）

**预计耗时**：7-10天

---

### 【阶段二】LLM Agent核心开发（2-3周）

#### 目标
实现能够"读代码"、"写代码"、"做反思"的智能体。

#### 任务清单

##### 任务2.1：LLM接口封装

**新增 `llm_reward_agent/agent/llm_interface.py`**：
```python
import openai
from typing import List, Dict

class LLMInterface:
    """统一的LLM调用接口，支持多种模型"""
    
    def __init__(self, model_name="gpt-4", api_key=None, base_url=None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
        """
        生成N个不同的回复
        
        Args:
            prompt: 提示词
            n: 生成数量
            temperature: 温度参数（越高越随机）
        
        Returns:
            List[str]: N个生成结果
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature
        )
        return [choice.message.content for choice in response.choices]
    
    def analyze(self, prompt: str) -> str:
        """单次分析调用（用于Reflection）"""
        return self.generate(prompt, n=1, temperature=0.3)[0]
```

**配置文件 `llm_reward_agent/config/llm_config.yaml`**：
```yaml
llm:
  provider: "openai"  # 可选：openai, anthropic, zhipu, etc.
  model: "gpt-4"
  api_key: "your-api-key-here"  # 或通过环境变量
  base_url: null  # 国产大模型可指定自定义URL
  
generation:
  num_candidates: 4  # 每代生成4个候选
  temperature: 0.8   # 高温度保证多样性
  max_tokens: 2000
  
reflection:
  temperature: 0.3   # 低温度保证分析准确性
  max_tokens: 1000
```

---

##### 任务2.2：提示词工程（Prompt Engineering）

这是**最核心**的部分！提示词质量直接决定LLM生成代码的质量。

**新增 `llm_reward_agent/agent/prompt_templates.py`**：

```python
class PromptTemplates:
    """LLM提示词模板库"""
    
    @staticmethod
    def initial_generation_prompt(env_context, task_description):
        """第一代生成（Zero-Shot）"""
        return f"""你是一位专业的强化学习奖励工程师。请根据以下信息设计奖励函数。

# 任务描述
{task_description}

# 环境信息
- 观测空间：{env_context['observation_space']}
- 动作空间：{env_context['action_space']}
- 物理参数：
{env_context['physical_constants']}

# 环境代码片段（关键部分）
```python
{env_context['code_snippet']}
```

# 要求
1. 实现 `compute_reward()` 函数，签名如下：
   def compute_reward(agent_name, observation, global_state, actions, world):
       # 你的代码
       return total_reward, components

2. `components` 必须是字典，包含各个奖励分量（用于日志分析），例如：
   {{'distance_reward': ..., 'collision_penalty': ..., 'formation_reward': ...}}

3. 考虑以下目标：
   - 鼓励智能体接近并围捕目标
   - 惩罚智能体之间的碰撞
   - 奖励形成均匀的包围圈
   - 适当惩罚能耗

4. 只输出Python代码，不要有任何额外解释。代码需要符合PEP8规范。

# 输出格式
```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    
    # 你的代码实现
    ...
    
    total_reward = sum(components.values())
    return total_reward, components
```
"""
    
    @staticmethod
    def evolution_prompt(env_context, task_description, parent_code, reflection):
        """进化生成（基于上一代的改进）"""
        return f"""你是一位专业的强化学习奖励工程师。基于上一代的训练反馈，改进奖励函数。

# 任务描述
{task_description}

# 环境信息
（与初始代相同，省略...）

# 上一代最优代码
```python
{parent_code}
```

# 训练反馈与反思
{reflection}

# 改进要求
1. 基于上述反思，对代码进行修改：
   - 调整权重系数
   - 增加/删除奖励项
   - 改变函数形式（线性 → 指数 / 阈值函数等）
   
2. 生成 **{n_candidates}** 个不同的变体（Mutation），每个变体应有明显差异。

3. 只输出Python代码，每个变体之间用 `# --- Variant {i} ---` 分隔。

# 输出格式
```python
# --- Variant 0 ---
def compute_reward(agent_name, observation, global_state, actions, world):
    ...

# --- Variant 1 ---
def compute_reward(agent_name, observation, global_state, actions, world):
    ...

... （以此类推）
```
"""
    
    @staticmethod
    def reflection_prompt(training_logs):
        """生成反思（Reward Reflection）"""
        return f"""你是一位专业的强化学习研究员。请分析以下训练日志，总结奖励函数的表现。

# 训练日志
{training_logs}

# 分析要求
1. **奖励分量诊断**：
   - 哪些分量起到了作用（数值非零且有方差）？
   - 哪些分量失效了（一直为0或数值异常）？
   - 各分量的权重是否合理？

2. **任务性能分析**：
   - 成功率是否达标？
   - 完成时间是否过长？
   - 是否存在明显的失败模式？

3. **协同行为评估**：
   - 智能体是否学会了均匀包围？
   - 是否出现碰撞或扎堆现象？

4. **改进建议**（针对下一代）：
   - 需要增加/删除哪些奖励项？
   - 需要调整哪些权重系数？
   - 需要改变哪些函数形式？

# 输出格式
使用自然语言，分点总结，不超过500字。
"""
```

---

##### 任务2.3：主Agent类实现

**新增 `llm_reward_agent/agent/reward_design_agent.py`**：

```python
import os
import json
from typing import List, Dict, Tuple
from .llm_interface import LLMInterface
from .prompt_templates import PromptTemplates
from .memory import EvolutionaryMemory
from ..tools.context_extractor import EnvironmentContextExtractor

class RewardDesignAgent:
    """奖励函数设计智能体"""
    
    def __init__(self, config_path="config/llm_config.yaml"):
        # 1. 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. 初始化组件
        self.llm = LLMInterface(
            model_name=self.config['llm']['model'],
            api_key=self.config['llm']['api_key']
        )
        self.memory = EvolutionaryMemory()
        self.context_extractor = EnvironmentContextExtractor()
        self.prompt_builder = PromptTemplates()
    
    def initialize(self, env_file_path: str, task_description: str):
        """初始化：读取环境代码"""
        self.env_context = self.context_extractor.extract_skeleton(env_file_path)
        self.task_description = task_description
        print("✅ 环境上下文已提取")
    
    def generate_candidates(self, generation: int) -> List[str]:
        """生成候选奖励函数代码"""
        if generation == 0:
            # 第一代：Zero-Shot生成
            prompt = self.prompt_builder.initial_generation_prompt(
                self.env_context, 
                self.task_description
            )
        else:
            # 后续代：基于父本进化
            parent_code = self.memory.get_best_code(generation - 1)
            reflection = self.memory.get_reflection(generation - 1)
            prompt = self.prompt_builder.evolution_prompt(
                self.env_context,
                self.task_description,
                parent_code,
                reflection
            )
        
        # 调用LLM生成
        n_candidates = self.config['generation']['num_candidates']
        raw_outputs = self.llm.generate(
            prompt, 
            n=n_candidates, 
            temperature=self.config['generation']['temperature']
        )
        
        # 解析代码（可能需要正则提取```python...```块）
        codes = self._parse_code_blocks(raw_outputs)
        
        return codes
    
    def analyze_results(self, results: List[Dict]) -> Tuple[str, str]:
        """分析训练结果，生成反思"""
        # 1. 找到最优代码
        best_result = max(results, key=lambda x: x['fitness'])
        
        # 2. 构建日志摘要
        logs_summary = self._format_logs(results)
        
        # 3. 调用LLM生成Reflection
        prompt = self.prompt_builder.reflection_prompt(logs_summary)
        reflection = self.llm.analyze(prompt)
        
        return best_result['code'], reflection
    
    def step(self, generation: int) -> Dict:
        """执行一代进化（核心流程）"""
        print(f"\n{'='*60}")
        print(f"🧬 Generation {generation}")
        print(f"{'='*60}")
        
        # 1. 生成候选代码
        print("🤖 LLM正在生成候选奖励函数...")
        codes = self.generate_candidates(generation)
        print(f"✅ 生成了 {len(codes)} 个候选")
        
        # 2. 调用仿真工具（见任务3）
        from ..tools.simulation_tool import SimulationTool
        simulator = SimulationTool()
        results = simulator.run_parallel(codes, generation)
        
        # 3. 分析结果
        print("🔍 LLM正在分析训练日志...")
        best_code, reflection = self.analyze_results(results)
        
        # 4. 更新记忆
        self.memory.save(generation, best_code, reflection, results)
        
        return {
            'generation': generation,
            'best_code': best_code,
            'best_fitness': max(r['fitness'] for r in results),
            'reflection': reflection
        }
    
    def _parse_code_blocks(self, raw_outputs: List[str]) -> List[str]:
        """从LLM输出中提取Python代码块"""
        import re
        codes = []
        for output in raw_outputs:
            # 匹配 ```python ... ```
            matches = re.findall(r'```python\n(.*?)\n```', output, re.DOTALL)
            if matches:
                codes.extend(matches)
            else:
                # 如果没有代码块标记，尝试直接提取
                codes.append(output)
        return codes
    
    def _format_logs(self, results: List[Dict]) -> str:
        """格式化日志为自然语言"""
        summary = "# 本代4个候选的训练结果\n\n"
        for i, r in enumerate(results):
            summary += f"## Candidate {i}\n"
            summary += f"- 成功率：{r['metrics']['success_rate']:.2%}\n"
            summary += f"- 平均围捕时间：{r['metrics']['avg_capture_time']:.1f} steps\n"
            summary += f"- 奖励分量统计：\n"
            for key, val in r['metrics']['reward_components'].items():
                summary += f"  * {key}: {val:.4f}\n"
            summary += "\n"
        return summary
```

---

##### 任务2.4：进化记忆管理

**新增 `llm_reward_agent/agent/memory.py`**：
```python
import json
import os
from typing import Dict, List

class EvolutionaryMemory:
    """进化历史记忆库"""
    
    def __init__(self, save_dir="experiments/evolution_archive"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = []  # [{generation, best_code, reflection, fitness}]
    
    def save(self, generation: int, code: str, reflection: str, all_results: List[Dict]):
        """保存一代的记录"""
        record = {
            'generation': generation,
            'best_code': code,
            'reflection': reflection,
            'best_fitness': max(r['fitness'] for r in all_results),
            'all_results': all_results
        }
        self.history.append(record)
        
        # 持久化到文件
        filepath = os.path.join(self.save_dir, f"generation_{generation}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    
    def get_best_code(self, generation: int) -> str:
        """获取指定代的最优代码"""
        return self.history[generation]['best_code']
    
    def get_reflection(self, generation: int) -> str:
        """获取指定代的反思"""
        return self.history[generation]['reflection']
    
    def get_best_ever(self) -> Dict:
        """获取历史最优"""
        return max(self.history, key=lambda x: x['best_fitness'])
```

---

#### 阶段二验收标准

- [x] LLM可以根据环境代码生成奖励函数（手动测试：给一个简化的环境描述）
- [x] 生成的代码语法正确（通过 `ast.parse()` 检查）
- [x] 反思生成合理（人工评估：是否抓住了关键问题）

**预计耗时**：10-15天

---

### 【阶段三】并行训练框架（1-2周）

#### 目标
实现高效的"生成4个代码 → 同时跑4个训练 → 收集4份日志"流程。

#### 任务清单

##### 任务3.1：沙盒管理器

**新增 `llm_reward_agent/tools/sandbox_manager.py`**：
```python
import os
import shutil
from typing import List

class SandboxManager:
    """管理并行训练的沙盒目录"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
    
    def create_sandboxes(self, generation: int, codes: List[str]) -> List[str]:
        """
        为每个候选代码创建独立的沙盒
        
        Returns:
            List[str]: 沙盒目录路径列表
        """
        gen_dir = os.path.join(self.base_dir, f"generation_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        
        sandbox_paths = []
        for i, code in enumerate(codes):
            sandbox_path = os.path.join(gen_dir, f"candidate_{i}")
            os.makedirs(sandbox_path, exist_ok=True)
            
            # 1. 复制基座代码（环境、训练脚本等）
            self._setup_base_code(sandbox_path)
            
            # 2. 写入生成的奖励函数
            reward_file = os.path.join(sandbox_path, "MADDPG/envs/reward_function.py")
            with open(reward_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            sandbox_paths.append(sandbox_path)
        
        return sandbox_paths
    
    def _setup_base_code(self, sandbox_path):
        """设置基础代码（通过软链接或复制）"""
        # 方案1：软链接（推荐，节省空间）
        src_maddpg = os.path.abspath("MADDPG")
        dst_maddpg = os.path.join(sandbox_path, "MADDPG")
        
        if os.name == 'nt':  # Windows
            # Windows不支持软链接，只能复制
            shutil.copytree(src_maddpg, dst_maddpg, dirs_exist_ok=True)
        else:  # Linux/Mac
            os.symlink(src_maddpg, dst_maddpg, target_is_directory=True)
        
        # 方案2：只复制必需文件
        # ...
```

---

##### 任务3.2：并行调度器

**新增 `launcher.py`**（项目根目录）：
```python
import subprocess
import time
import os
from multiprocessing import Pool
from typing import List, Dict

class ParallelLauncher:
    """并行训练任务调度器"""
    
    def __init__(self, max_workers=4, timeout=1200):
        """
        Args:
            max_workers: 最大并行数（取决于CPU核心数/GPU数量）
            timeout: 单个训练超时时间（秒）
        """
        self.max_workers = max_workers
        self.timeout = timeout
    
    def run_parallel(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        并行执行训练任务
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 每个候选的训练结果
        """
        print(f"🚀 启动 {len(sandbox_paths)} 个并行训练任务...")
        
        with Pool(processes=self.max_workers) as pool:
            results = pool.map(self._run_single_training, sandbox_paths)
        
        print("✅ 所有训练任务完成")
        return results
    
    def _run_single_training(self, sandbox_path: str) -> Dict:
        """执行单个训练任务（子进程）"""
        candidate_id = os.path.basename(sandbox_path)
        print(f"  [{candidate_id}] 开始训练...")
        
        # 1. 组装命令
        cmd = [
            "python", "MADDPG/main_train.py",
            "--env_name", "simple_tag_env",
            "--episode_num", "100",  # 【关键】轻量化训练，只跑100轮
            "--episode_length", "100",
            "--render_mode", "None"
        ]
        
        # 2. 执行（捕获输出，避免屏幕混乱）
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=sandbox_path,  # 【关键】在沙盒目录中执行
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  [{candidate_id}] 训练完成 ({elapsed:.1f}s)")
                # 3. 读取日志
                metrics = self._parse_logs(sandbox_path)
                return {
                    'id': candidate_id,
                    'status': 'success',
                    'metrics': metrics,
                    'elapsed': elapsed
                }
            else:
                print(f"  [{candidate_id}] 训练失败: {result.stderr[-200:]}")
                return {'id': candidate_id, 'status': 'error', 'error': result.stderr}
        
        except subprocess.TimeoutExpired:
            print(f"  [{candidate_id}] 训练超时")
            return {'id': candidate_id, 'status': 'timeout'}
        
        except Exception as e:
            print(f"  [{candidate_id}] 未知错误: {e}")
            return {'id': candidate_id, 'status': 'error', 'error': str(e)}
    
    def _parse_logs(self, sandbox_path: str) -> Dict:
        """解析训练日志，提取性能指标"""
        log_file = os.path.join(sandbox_path, "MADDPG/logs/reward_component_stats.json")
        
        if not os.path.exists(log_file):
            return {'fitness': 0.0}
        
        with open(log_file, 'r') as f:
            stats = json.load(f)
        
        # 计算综合Fitness（加权组合多个指标）
        success_rate = stats['task_performance']['success_rate']
        avg_time = stats['task_performance']['avg_capture_time']
        
        # Fitness = 成功率 - 时间惩罚
        fitness = success_rate - 0.001 * avg_time
        
        return {
            'fitness': fitness,
            'success_rate': success_rate,
            'avg_capture_time': avg_time,
            'reward_components': stats['reward_components']
        }

if __name__ == "__main__":
    # 测试：手动创建4个沙盒并运行
    sandboxes = ["experiments/test/candidate_0", "experiments/test/candidate_1", ...]
    launcher = ParallelLauncher(max_workers=4)
    results = launcher.run_parallel(sandboxes)
    print(results)
```

---

##### 任务3.3：仿真工具集成

**新增 `llm_reward_agent/tools/simulation_tool.py`**：
```python
from .sandbox_manager import SandboxManager
from ..launcher import ParallelLauncher  # 使用launcher.py

class SimulationTool:
    """Agent的仿真执行工具"""
    
    def __init__(self):
        self.sandbox_mgr = SandboxManager()
        self.launcher = ParallelLauncher(max_workers=4, timeout=1200)
    
    def run_parallel(self, codes: List[str], generation: int) -> List[Dict]:
        """
        并行运行多个候选代码
        
        Args:
            codes: LLM生成的代码列表
            generation: 当前代数
        
        Returns:
            List[Dict]: 训练结果（包含fitness、metrics等）
        """
        # 1. 创建沙盒
        print("📦 创建训练沙盒...")
        sandbox_paths = self.sandbox_mgr.create_sandboxes(generation, codes)
        
        # 2. 并行执行
        results = self.launcher.run_parallel(sandbox_paths)
        
        # 3. 为每个结果附加对应的代码
        for i, result in enumerate(results):
            result['code'] = codes[i]
        
        return results
```

---

#### 阶段三验收标准

- [x] 能够并行运行4个训练任务（手动创建4个不同的reward_function.py并测试）
- [x] 日志正确保存到各自的沙盒目录
- [x] Fitness计算合理（人工检查：高性能代码的fitness应该更高）

**预计耗时**：7-10天

---

### 【阶段四】反馈闭环与整合（1周）

#### 目标
将所有组件串联成完整的进化循环。

#### 任务清单

##### 任务4.1：主流程脚本

**新增 `run_evolution.py`**（项目根目录）：
```python
import argparse
from llm_reward_agent.agent.reward_design_agent import RewardDesignAgent

def main(args):
    # 1. 初始化Agent
    agent = RewardDesignAgent(config_path=args.config)
    agent.initialize(
        env_file_path="MADDPG/envs/simple_tag_env.py",
        task_description="""
        任务：3个追捕智能体协同围捕1个逃逸目标。
        要求：
        1. 追捕者需要接近并包围目标
        2. 追捕者之间避免碰撞
        3. 形成均匀的包围圈
        4. 在尽可能短的时间内完成围捕
        """
    )
    
    # 2. 进化循环
    for generation in range(args.num_generations):
        result = agent.step(generation)
        
        print(f"\n📊 Generation {generation} 最优结果：")
        print(f"  Fitness: {result['best_fitness']:.4f}")
        print(f"  Reflection: {result['reflection'][:200]}...")
        
        # 保存到文件
        with open(f"experiments/generation_{generation}_summary.txt", 'w') as f:
            f.write(f"Best Code:\n{result['best_code']}\n\n")
            f.write(f"Reflection:\n{result['reflection']}\n")
    
    # 3. 输出最终结果
    best_ever = agent.memory.get_best_ever()
    print(f"\n🏆 进化完成！历史最优出现在第 {best_ever['generation']} 代")
    print(f"   Fitness: {best_ever['best_fitness']:.4f}")
    
    # 保存最终代码到MADDPG目录
    final_code_path = "MADDPG/envs/reward_function_final.py"
    with open(final_code_path, 'w') as f:
        f.write(best_ever['best_code'])
    print(f"✅ 最优奖励函数已保存到：{final_code_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='llm_reward_agent/config/llm_config.yaml')
    parser.add_argument('--num_generations', type=int, default=5, help='进化代数')
    args = parser.parse_args()
    
    main(args)
```

**运行示例**：
```bash
python run_evolution.py --num_generations 5
```

---

##### 任务4.2：错误处理与容错

**增强 `reward_design_agent.py`**：
```python
def generate_candidates(self, generation: int) -> List[str]:
    """生成候选代码（增加错误处理）"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            codes = self._generate_candidates_unsafe(generation)
            
            # 语法检查
            valid_codes = []
            for code in codes:
                if self._syntax_check(code):
                    valid_codes.append(code)
                else:
                    print(f"⚠️ 跳过语法错误的代码")
            
            if len(valid_codes) >= 2:  # 至少保证2个有效候选
                return valid_codes
            else:
                print(f"⚠️ 有效代码不足，重新生成（第{attempt+1}次尝试）")
        
        except Exception as e:
            print(f"❌ 生成失败：{e}")
    
    # 失败后使用后备方案：返回上一代的代码
    print("⚠️ LLM生成失败，使用上一代代码作为后备")
    if generation > 0:
        return [self.memory.get_best_code(generation - 1)]
    else:
        # 第0代失败，使用人工基准
        return [self._get_human_baseline()]

def _syntax_check(self, code: str) -> bool:
    """检查代码语法是否正确"""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def _get_human_baseline(self) -> str:
    """返回人工设计的基准奖励函数"""
    with open("reward_templates/base_reward.py", 'r') as f:
        return f.read()
```

---

#### 阶段四验收标准

- [x] 能够完整运行5代进化（可以使用降低的训练轮数，如每个候选只训练50轮）
- [x] 每代都有保存记录（JSON文件）
- [x] 出现语法错误时能够自动重试或使用后备方案

**预计耗时**：5-7天

---

### 【阶段五】实验与论文（2-3周）

#### 目标
产出可发表的研究成果。

#### 任务清单

##### 任务5.1：完整实验

**实验设计**：
```
1. 基准实验（Baseline）
   - 人工设计的奖励函数（3-5个不同版本）
   - 训练5000轮，记录成功率、时间等

2. LLM实验（Main）
   - 运行10代进化（每代4个候选）
   - 对比不同LLM（GPT-4, Claude-3.5, DeepSeek等）
   - 消融实验：
     * 无Reflection vs 有Reflection
     * 不同进化代数的影响

3. 对比分析
   - 最优LLM设计 vs 最优人工设计
   - 收敛速度分析
   - Token消耗与成本分析
```

**数据记录表**：
| 方法 | 成功率 | 平均时间 | 碰撞率 | 围捕角度方差 |
|------|--------|---------|--------|-------------|
| 人工设计v1 | 0.72 | 52.3 | 0.15 | 0.34 |
| 人工设计v2 | 0.78 | 48.1 | 0.12 | 0.28 |
| LLM（第1代）| 0.65 | 58.7 | 0.20 | 0.41 |
| LLM（第5代）| 0.82 | 45.2 | 0.08 | 0.22 |
| LLM（第10代）| 0.89 | 42.5 | 0.05 | 0.18 |

---

##### 任务5.2：可视化

**新增 `visualization/evolution_plot.py`**：
```python
import matplotlib.pyplot as plt
import json
import os

def plot_evolution_curve():
    """绘制进化曲线"""
    generations = []
    best_fitness = []
    avg_fitness = []
    
    for i in range(10):
        with open(f"experiments/evolution_archive/generation_{i}.json") as f:
            data = json.load(f)
            generations.append(i)
            best_fitness.append(data['best_fitness'])
            avg_fitness.append(np.mean([r['fitness'] for r in data['all_results']]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'o-', label='Best Fitness', linewidth=2)
    plt.plot(generations, avg_fitness, 's--', label='Average Fitness', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolution Curve of Reward Function Design', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/evolution_curve.png', dpi=300)
    plt.show()

def plot_reward_components_comparison():
    """对比不同方法的奖励分量"""
    # 读取人工设计 vs LLM设计的日志
    # 绘制雷达图或柱状图
    ...
```

---

##### 任务5.3：论文撰写

**论文结构**（以中文毕业论文为例）：

```
第1章 绪论
  1.1 研究背景与意义
  1.2 国内外研究现状
      1.2.1 多智能体强化学习
      1.2.2 奖励函数设计方法
      1.2.3 大语言模型在RL中的应用
  1.3 论文主要工作与创新点
  1.4 论文组织结构

第2章 相关理论与技术基础
  2.1 多智能体强化学习
      2.1.1 MARL基本概念
      2.1.2 MADDPG算法原理
  2.2 奖励函数工程
      2.2.1 奖励函数设计挑战
      2.2.2 传统设计方法
  2.3 大语言模型
      2.3.1 LLM基本原理
      2.3.2 代码生成能力
      2.3.3 Agent架构设计
  2.4 EUREKA框架介绍
  2.5 本章小结

第3章 系统总体设计
  3.1 系统架构设计
  3.2 问题建模
      3.2.1 多智能体围捕任务定义
      3.2.2 奖励函数设计问题形式化
  3.3 LLM-Agent设计
      3.3.1 感知模块
      3.3.2 推理模块
      3.3.3 行动模块
      3.3.4 记忆模块
  3.4 进化算法设计
  3.5 本章小结

第4章 关键技术实现
  4.1 环境接口标准化
  4.2 提示词工程
      4.2.1 初始生成提示词设计
      4.2.2 进化提示词设计
      4.2.3 反思提示词设计
  4.3 并行训练框架
      4.3.1 沙盒隔离机制
      4.3.2 资源调度策略
  4.4 日志分析与反馈
      4.4.1 多维度指标设计
      4.4.2 奖励分量诊断方法
  4.5 本章小结

第5章 实验与结果分析
  5.1 实验环境与参数设置
  5.2 基准实验
      5.2.1 人工设计的奖励函数
      5.2.2 训练结果分析
  5.3 LLM实验
      5.3.1 进化过程分析
      5.3.2 不同LLM对比
      5.3.3 消融实验
  5.4 对比分析
      5.4.1 性能对比
      5.4.2 成本分析
      5.4.3 可解释性分析
  5.5 案例研究
      5.5.1 典型进化路径
      5.5.2 失败案例分析
  5.6 本章小结

第6章 总结与展望
  6.1 论文工作总结
  6.2 主要创新点
  6.3 不足与展望

参考文献
致谢
附录A 关键代码
附录B 完整进化日志
```

---

#### 阶段五交付物

- [x] 完整实验数据（CSV/JSON）
- [x] 可视化图表（进化曲线、对比图等）
- [x] 论文初稿（中文）
- [x] 会议论文投稿版（英文，可选）
- [x] 答辩PPT

**预计耗时**：15-20天

---

## 4. 关键技术实现细节

### 4.1 提示词工程技巧

**技巧1：提供少样本示例（Few-Shot Learning）**
```python
# 在提示词中加入2-3个示例代码
example_1 = """
# 示例1：简单距离奖励
def compute_reward(...):
    dist = np.linalg.norm(agent_pos - target_pos)
    return -dist, {'distance_reward': -dist}
"""

prompt = f"""参考以下示例设计奖励函数：
{example_1}

现在请为以下任务设计...
"""
```

**技巧2：约束输出格式**
```python
# 要求LLM输出JSON格式的代码
prompt += """
输出格式（严格遵守）：
{
  "code": "def compute_reward(...): ...",
  "explanation": "设计思路说明",
  "key_components": ["distance_reward", "collision_penalty"]
}
"""
```

**技巧3：链式推理（Chain-of-Thought）**
```python
prompt += """
请按以下步骤思考：
1. 分析任务的核心目标是什么？
2. 有哪些约束条件（安全、效率）？
3. 需要哪些奖励分量来引导行为？
4. 如何设置权重系数？

然后再编写代码。
"""
```

---

### 4.2 并行训练优化

**优化1：共享基座代码（减少磁盘占用）**
```bash
# 使用硬链接而非复制
ln -s $(pwd)/MADDPG sandbox_0/MADDPG
```

**优化2：GPU显存管理**
```python
# 在train.py中限制每个进程的显存使用
import torch
torch.cuda.set_per_process_memory_fraction(0.25)  # 4个进程各占25%
```

**优化3：提前终止（Early Stopping）**
```python
# 如果连续10轮没有提升，提前结束训练
if no_improvement_count > 10:
    break
```

---

### 4.3 日志分析技巧

**技巧1：异常检测**
```python
def detect_anomalies(reward_components):
    """检测奖励函数是否异常"""
    issues = []
    
    # 检查1：某个分量一直为0
    for key, values in reward_components.items():
        if np.std(values) < 1e-6:
            issues.append(f"{key}分量无方差，可能未触发")
    
    # 检查2：某个分量过大
    for key, values in reward_components.items():
        if np.abs(np.mean(values)) > 100:
            issues.append(f"{key}分量数值过大，可能主导训练")
    
    return issues
```

**技巧2：因果分析**
```python
# 计算奖励分量与成功率的相关性
from scipy.stats import pearsonr

for component in ['distance_reward', 'collision_penalty', ...]:
    corr, p_value = pearsonr(
        reward_data[component],
        success_rate_data
    )
    print(f"{component} 与成功率的相关性：{corr:.3f} (p={p_value:.3f})")
```

---

## 5. 风险评估与应对

### 5.1 技术风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|-------|------|---------|
| LLM生成的代码语法错误 | 高 | 中 | 增加语法检查与重试机制 |
| 训练不收敛 | 中 | 高 | 降低学习率、增加训练轮数 |
| GPU/CPU资源不足 | 中 | 中 | 减少并行数、使用云GPU |
| LLM API不稳定 | 中 | 中 | 增加重试机制、备用API |
| 进化陷入局部最优 | 中 | 中 | 增加Temperature、定期注入随机代码 |

### 5.2 进度风险

**风险1：开发时间超期**
- **应对**：每周设置里程碑检查点，及时调整范围
- **应对**：优先保证核心功能，可视化等次要功能可延后

**风险2：实验结果不理想**
- **应对**：准备多个Baseline对比
- **应对**：即使性能不如人工，也可强调"自动化"的价值

---

## 6. 质量保障与测试

### 6.1 单元测试

**测试 `reward_function.py`**：
```python
def test_reward_function():
    """测试奖励函数接口正确性"""
    from MADDPG.envs.reward_function import compute_reward
    
    # 构造假数据
    global_state = {
        'agent_positions': np.array([[0, 0], [1, 0], [0, 1]]),
        'prey_position': np.array([0.5, 0.5]),
        ...
    }
    
    reward, components = compute_reward('adversary_0', None, global_state, {}, None)
    
    # 断言
    assert isinstance(reward, float)
    assert isinstance(components, dict)
    assert 'distance_reward' in components
```

### 6.2 集成测试

**测试完整流程**：
```python
def test_full_pipeline():
    """测试从生成到训练的完整流程"""
    # 1. 生成假代码
    code = "def compute_reward(...): return 0.0, {}"
    
    # 2. 创建沙盒
    sandbox_mgr = SandboxManager()
    paths = sandbox_mgr.create_sandboxes(0, [code])
    
    # 3. 运行训练
    launcher = ParallelLauncher(max_workers=1, timeout=60)
    results = launcher.run_parallel(paths)
    
    # 4. 验证
    assert len(results) == 1
    assert results[0]['status'] == 'success'
```

### 6.3 验收标准

**最终验收清单**：
- [ ] 能够完整运行10代进化（每代4个候选）
- [ ] LLM设计的奖励函数在某些指标上优于人工基准
- [ ] 系统能够自动从失败中恢复（错误重试、后备方案）
- [ ] 代码有完善的文档和注释
- [ ] 论文初稿完成（包含所有章节）

---

## 7. 论文撰写指南

### 7.1 创新点提炼

**创新点1：方法论创新**
> 首次将EUREKA框架从单智能体扩展到多智能体协同任务，设计了针对MARL的进化式奖励函数生成方法。

**创新点2：指标设计**
> 提出了围捕任务的协同行为评估指标（围捕角度方差、编队质量等），为LLM反思提供了关键反馈信号。

**创新点3：工程实现**
> 设计了沙盒并行训练框架，实现了候选奖励函数的高效验证，解决了LLM-MARL集成的工程化挑战。

### 7.2 实验设计原则

1. **对照组设计**：
   - 人工设计（多个版本）
   - 随机生成（证明LLM的价值）
   - 固定权重vs学习权重

2. **消融实验**：
   - 有/无Reflection机制
   - 不同进化代数
   - 不同LLM模型

3. **可视化**：
   - 进化曲线（必须）
   - 训练曲线对比（必须）
   - 包围圈可视化（加分项）
   - 权重热力图（加分项）

### 7.3 常见问题预案

**Q1：审稿人质疑"为什么LLM比人工好"？**
- **答**：强调不是一定"更好"，而是"自动化"，节省人力
- **答**：展示LLM发现了人类没想到的设计（如特殊的非线性函数）

**Q2：审稿人质疑"训练轮数太少，不够收敛"？**
- **答**：说明是轻量化验证，目的是快速筛选，最优版本会完整训练
- **答**：补充实验：用最优代码跑完整训练

**Q3：审稿人质疑"只在一个任务上测试"？**
- **答**：时间有限，优先保证深度
- **答**：讨论章节提到可扩展性，列出未来工作

---

## 8. 附录：核心代码模板

### 8.1 奖励函数模板

**模板1：线性组合**
```python
def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    
    # 权重系数
    w_dist = 1.0
    w_coll = -5.0
    w_form = 0.5
    
    # 分量计算
    components['distance'] = -get_distance_to_prey(...)
    components['collision'] = -check_collision(...)
    components['formation'] = calc_formation_quality(...)
    
    total = w_dist * components['distance'] + w_coll * components['collision'] + w_form * components['formation']
    return total, components
```

**模板2：分段函数**
```python
def compute_reward(...):
    dist = get_distance_to_prey(...)
    
    if dist < 0.5:  # 已接近
        reward = 10.0  # 大奖励
    elif dist < 2.0:  # 中等距离
        reward = -0.1 * dist
    else:  # 太远
        reward = -1.0
    
    return reward, {'distance_reward': reward}
```

**模板3：势场法（Potential Field）**
```python
def compute_reward(...):
    # 吸引势场（引导接近）
    attraction = -k_att * distance_to_prey ** 2
    
    # 排斥势场（避免碰撞）
    repulsion = 0
    for teammate_dist in inter_agent_distances:
        if teammate_dist < d_safe:
            repulsion += k_rep / teammate_dist ** 2
    
    components = {'attraction': attraction, 'repulsion': repulsion}
    return attraction + repulsion, components
```

---

### 8.2 评估脚本模板

**新增 `MADDPG/main_evaluate_with_reward_analysis.py`**：
```python
"""评估特定奖励函数的性能"""

from main_parameters import main_parameters
from envs import simple_tag_env
from agents.maddpg.MADDPG_agent import MADDPG
import numpy as np

def evaluate_reward_function(reward_file_path, num_episodes=100):
    """
    评估指定奖励函数的性能
    
    Returns:
        Dict: {
            'success_rate': float,
            'avg_capture_time': float,
            'reward_components': Dict[str, float]
        }
    """
    # 1. 替换奖励函数
    import shutil
    shutil.copy(reward_file_path, "envs/reward_function.py")
    
    # 2. 加载环境和智能体
    args = main_parameters()
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length)
    agent = MADDPG(...)
    agent.load_model()  # 加载预训练模型
    
    # 3. 运行测试
    successes = []
    capture_times = []
    all_components = {key: [] for key in ['distance', 'collision', 'formation']}
    
    for ep in range(num_episodes):
        obs = env.reset()
        for step in range(args.episode_length):
            actions = agent.get_actions(obs, eval=True)
            obs, rewards, dones, infos = env.step(actions)
            
            # 收集奖励分量
            for key in all_components.keys():
                all_components[key].append(env.last_reward_components[...][key])
            
            if check_capture_success(env):
                successes.append(1)
                capture_times.append(step)
                break
        else:
            successes.append(0)
    
    # 4. 统计
    return {
        'success_rate': np.mean(successes),
        'avg_capture_time': np.mean(capture_times) if capture_times else np.inf,
        'reward_components': {k: np.mean(v) for k, v in all_components.items()}
    }

if __name__ == "__main__":
    # 对比多个奖励函数
    results = {}
    for name, path in [
        ("Human_v1", "reward_templates/human_v1.py"),
        ("LLM_gen0", "experiments/generation_0/candidate_0/reward_function.py"),
        ("LLM_gen5", "experiments/generation_5/candidate_0/reward_function.py"),
    ]:
        results[name] = evaluate_reward_function(path)
    
    # 打印对比表
    print("方法\t\t成功率\t平均时间")
    for name, metrics in results.items():
        print(f"{name}\t{metrics['success_rate']:.2%}\t{metrics['avg_capture_time']:.1f}")
```

---

## 9. 时间线与里程碑

### 甘特图（文本版）

```
Week 1-2:  [====================] 阶段一：环境接口标准化
Week 3-4:  [=====>              ] 阶段二：LLM Agent开发（前半）
Week 5:    [        ====>       ] 阶段二：LLM Agent开发（后半）
Week 6:    [             ===>   ] 阶段三：并行训练框架
Week 7:    [                 ==>] 阶段四：反馈闭环
Week 8-9:  [====================] 阶段五：实验（前半）
Week 10:   [          ==>       ] 阶段五：论文撰写
Week 11:   [             =====>] 缓冲周（论文润色、答辩准备）
```

### 关键里程碑

| 日期 | 里程碑 | 交付物 |
|------|--------|--------|
| Week 2末 | 阶段一完成 | 可替换的奖励函数模块 + 增强日志 |
| Week 5末 | 阶段二完成 | LLM可生成合法代码 + 反思功能 |
| Week 7初 | 阶段三完成 | 并行训练框架可运行 |
| Week 7末 | 阶段四完成 | 完整进化循环可运行 |
| Week 9末 | 实验数据收集完毕 | 所有实验数据 + 可视化图表 |
| Week 10末 | 论文初稿完成 | 完整论文（待导师审阅） |
| Week 11末 | 答辩准备完成 | 最终论文 + PPT |

---

## 10. 总结

本实施计划提供了一份**详尽的、可操作的**开发蓝图，涵盖了从环境搭建到论文撰写的全流程。

### 核心要点回顾

1. **技术路线清晰**：五阶段递进式开发，每个阶段都有明确的验收标准
2. **风险可控**：识别了主要风险并提供应对预案
3. **工期合理**：总计7-11周，符合本科/硕士毕设时间
4. **可落地性强**：提供了大量代码模板和实现细节

### 成功的关键

- **提示词工程**：这是LLM能否生成高质量代码的核心
- **日志设计**：详细的反馈信号是进化的基础
- **实验充分**：对比实验要全面（基准、消融、不同LLM）

### 下一步行动

1. **立即开始**：按照阶段一的任务清单，先完成奖励函数解耦
2. **申请资源**：如需GPU，尽早申请实验室资源或云服务器
3. **定期汇报**：每周向导师汇报进度，及时调整计划

---

<div align="center">

**祝你毕设顺利！有任何问题欢迎随时讨论。**

📧 技术问题请提Issue | 📚 详细文档见README.md

</div>
