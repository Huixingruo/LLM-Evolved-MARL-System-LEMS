# 功能说明：初始奖励函数生成 — 思维链（CoT）两阶段管线

> 所属模块：`llm_reward_agent.agent`
> 涉及文件：`prompt_templates.py`、`reward_design_agent.py`
> 引入版本：v1.1
> 日期：2026-04-07

---

## 1. 背景与动机

### 问题诊断

在引入 CoT 之前，第一代（generation == 0）奖励函数采用**零样本（Zero-Shot）**生成策略：LLM 在接收到任务描述和环境代码片段后，直接输出 `compute_reward` 函数实现。

这种做法的根本缺陷在于：LLM 在**未对 Dec-POMDP 马尔可夫决策过程建立状态表征**的情况下，直接拼凑奖励项（如距离奖励、碰撞惩罚等），导致：

- **幻觉引用**：使用 `global_state` 中不存在的键名（如直接访问未定义的 `formation_score`）
- **物理语义错位**：不理解追捕者与逃跑者在速度/体积上的差异，导致奖励信号方向错误
- **包围圈表征缺失**：无法理解"什么样的空间拓扑代表高质量包围"，奖励函数缺乏对协同行为的引导

### 解决方案

强制实施**两阶段管线（Two-Stage Pipeline）**：先让 LLM 对环境进行**思维链（Chain of Thought, CoT）分析**，建立 MDP 状态表征；再以此为条件先验，高温度采样生成多样化候选代码。

---

## 2. 架构设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    generation == 0                        │
│                                                          │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  阶段一：CoT环境解析 │    │  阶段二：基于先验并行生成    │ │
│  │  llm.analyze()   │───▶│  llm.generate(n=3, T=0.9)  │ │
│  │  T=0.3 (低温度)   │    │                             │ │
│  └─────────────────┘    └─────────────────────────────┘ │
│         │                         │                      │
│         ▼                         ▼                      │
│  self.cot_analysis_result   候选代码列表                   │
│  (缓存，避免重复调用)         语法检查 ──▶ 有效候选          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 两阶段分工

| 阶段 | 目的 | LLM 调用方式 | Temperature | Token 上限 |
|------|------|------------|-------------|-----------|
| 阶段一 | 建立环境 MDP 表征 | `llm.analyze()` | 0.3（低温度，保证逻辑严密） | 1500 |
| 阶段二 | 生成候选奖励函数 | `llm.generate(n=3)` | `config_T + 0.2`（扩大探索宽度） | `max_tokens` |

---

## 3. 详细实现

### 3.1 `prompt_templates.py` — 新增方法

#### 方法 1：`cot_analysis_prompt`

```241:295:llm_reward_agent/agent/prompt_templates.py
    @staticmethod
    def cot_analysis_prompt(task_description: str, env_context: Dict) -> str:
```

**定位**：阶段一——强制 LLM 在生成代码前，按五个维度逐步分析环境。

**分析维度**：

| 维度 | 引导问题 |
|------|---------|
| 实现细节 (Implementation Details) | 环境代码使用了哪些依赖包？是否引入了外部未定义的变量？ |
| 环境结构 (Environment Structure) | `global_state` 包含哪些维度？观测向量如何拼接？ |
| 智能体交互 (Agent Interactions) | 捕食者与猎物的物理属性差异？碰撞判定的数学条件？ |
| 任务相关信息 (Task-relevant Information) | 核心目标与哪些变量挂钩？何种空间拓扑代表高质量包围圈？ |
| **API边界隔离 (API Boundaries)** | 上方代码片段中的 `CoreEnvLogic` 类**仅是文档伪代码**，绝非可在运行时被实例化的真实类。若需要物理常量，必须在函数内部以局部变量硬编码。 |

**约束**：LLM **只输出分析报告**，严格禁止生成任何奖励函数代码。

#### 方法 2：`initial_generation_prompt_with_cot`

```299:364:llm_reward_agent/agent/prompt_templates.py
    @staticmethod
    def initial_generation_prompt_with_cot(
        task_description: str,
        env_context: Dict,
        cot_analysis_result: str
    ) -> str:
```

**定位**：阶段二——以 CoT 分析结果为条件先验，生成 `compute_reward` 函数。

**关键约束**：
- 奖励分量必须保留在 `components` 字典中（供后续信用分配分析）
- 严格禁止输出任何解释性文字，只输出一个 ```python 代码块
- 强制过滤逃跑者：`if not global_state['is_adversary']: return 0.0, {}`
- **致命红线约束**：
  - **禁止实例化伪类**：绝对禁止在代码中写出 `CoreEnvLogic()`！代码片段仅是背景文档，运行时环境中根本不存在这个类。
  - **禁止虚构属性**：绝对禁止调用 `world.logic`、`world.adversary_params` 等不存在的属性。
  - **物理常量硬编码**：如果需要使用物理参数（如智能体的 `size=0.075`、地图大小 `world_size=2.5` 等），必须直接在函数内部以局部变量的形式硬编码声明（例如：`adv_size = 0.075`）。

### 3.2 `reward_design_agent.py` — 状态变量与管线

#### 状态变量

```68:llm_reward_agent/agent/reward_design_agent.py
        self.cot_analysis_result = None  # CoT思维链分析结果（两阶段管线用）
```

#### `initialize()` 中重置

```129:llm_reward_agent/agent/reward_design_agent.py
        # 每次初始化重置CoT缓存（确保新一代从头开始）
        self.cot_analysis_result = None
```

#### `generate_candidates()` 两阶段管线

```161:247:llm_reward_agent/agent/reward_design_agent.py
                if generation == 0:
                    # 阶段一：强制执行CoT环境解析（仅执行一次并缓存）
                    if not self.cot_analysis_result:
                        self.cot_analysis_result = self.llm.analyze(
                            prompt=analysis_prompt,
                            temperature=0.3,
                            max_tokens=1500
                        )
                    # 阶段二：基于CoT先验并行生成候选代码
                    raw_outputs = self.llm.generate(
                        prompt=generation_prompt,
                        n=n_candidates,
                        temperature=min(1.0, self.config['generation']['temperature'] + 0.2),
                        ...
                    )
```

**关键设计点**：

1. **缓存机制**：`cot_analysis_result` 在同一 `RewardDesignAgent` 实例生命周期内只调用一次，避免重复 API 开销
2. **温度差异化**：阶段一用低温（0.3）保证分析严谨，阶段二用较高温度（`config_T + 0.2`）扩大候选多样性
3. **渐进兼容**：对 `generation > 0` 的进化代保持原有逻辑不变（待后续任务二/三重构）

---

## 4. 使用说明

### 4.1 正常执行流程

当 `generation == 0` 时，控制台输出示例：

```
==================================================
🤖 第 0 代: 生成候选奖励函数
==================================================
🔍 [阶段一] 执行CoT环境与任务结构分析...
   ✅ CoT环境解析完成，已建立MDP表征先验。
📝 [阶段二] 基于先验并行生成代码... (尝试 1/3)
   ✅ 候选 0: 语法检查通过
   ✅ 候选 1: 语法检查通过
   ✅ 候选 2: 语法检查通过

✅ 成功生成 3 个有效候选
```

### 4.2 与后续任务的接口

CoT 分析结果会作为 `compute_reward` 生成的条件先验写入阶段二提示词中，但**不持久化到磁盘**（仅内存缓存）。后续任务二（多维收敛过滤拦截器）和任务三（EvoLeap 变异算子）可通过访问 `agent.cot_analysis_result` 获取该先验。

---

## 5. 已知约束与限制

| 约束 | 说明 |
|------|------|
| 仅第一代生效 | CoT 两阶段管线仅在 `generation == 0` 时触发，后续代保持进化逻辑 |
| API 依赖 | 依赖 `llm.analyze()` 接口；若 LLM 服务不可用，整个管线失败并触发后备方案 |
| Token 估算移除 | 移除了 `initialize()` 中的 Token 估算打印（阶段二 prompt 长度会动态变化） |
| 温度上限 | 第二阶段温度上限为 1.0，防止过度随机 |

---

## 6. 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `llm_reward_agent/agent/prompt_templates.py` | 新增方法 | `cot_analysis_prompt`、`initial_generation_prompt_with_cot` |
| `llm_reward_agent/agent/reward_design_agent.py` | 状态变量 + 重写方法 | 添加 `cot_analysis_result`，重写 `generate_candidates` |
