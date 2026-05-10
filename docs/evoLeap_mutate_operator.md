# 功能说明：DREAM 模块 — 诊断反思驱动的自适应进化算子

> **模块名称**：DREAM（Diagnostic Reflection-driven Evolutionary Adaptive Mutator）
> **中文释义**：诊断反思驱动的自适应进化算子
> **所属模块**：`llm_reward_agent.agent`
> **涉及文件**：`prompt_templates.py`、`reward_design_agent.py`
> **引入版本**：v2.0
> **日期**：2026-04-13
> **升级自**：EvoLeap 四向变异算子（v1.2）

---

## 1. 背景与动机

### 问题诊断（EvoLeap v1.2 的局限性）

在 EvoLeap v1.2 中，算子分配采用**静态正交波束搜索**：四个候选按固定顺序循环分配 `['F1', 'F2', 'F3', 'L1']`。这种做法的根本缺陷在于：

- **F2 算子效果不稳定**：F2（修剪精炼）直接删除失效分量的策略在实践中经常产生负向作用，因为失效分量往往是因为数学表达形式不对，而非该物理维度多余
- **算子分配缺乏适应性**：静态分配无法根据训练结果动态调整，当某些方向连续失败时仍会重复分配
- **探索效率低下**：每种算子固定分配一次，可能浪费计算资源在无望成功的方向上

### 解决方案（DREAM v2.0）

将静态正交波束搜索升级为**自适应算子分配**：

- **诊断反思升级**：Reflection 阶段从"仅输出病症诊断"升级为"诊断 + 算子分配方案"
- **自适应分配**：让 LLM 根据训练结果自主选择下一代的算子组合（每种最多 2 次）
- **F2 算子重构**：将"删除失效分量"改为"重构失效分量的数学表达"，保留物理维度信息
- **基数约束校验**：代码层强制校验（每种算子 ≤ 2 次），防止 LLM 输出格式错误

---

## 2. 架构设计

### 2.1 整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    generation > 0                                   │
│                                                                    │
│  ┌───────────────────┐     ┌─────────────────────────────────┐  │
│  │  获取父代数据       │     │  DREAM 自适应算子分配             │  │
│  │  parent_code       │────▶│  从 reflection 中解析算子分配      │  │
│  │  reflection        │     │  基数约束校验 (每种 ≤ 2次)        │  │
│  └───────────────────┘     └─────────────────────────────────┘  │
│                                    │                               │
│                                    ▼                               │
│                         ┌──────────────────────────────────┐   │
│                         │  ThreadPoolExecutor 并发调用        │   │
│                         │  ┌────┐ ┌────┐ ┌────┐ ┌────┐    │   │
│                         │  │F1  │ │F2* │ │F3  │ │L1  │    │   │
│                         │  └────┘ └────┘ └────┘ └────┘    │   │
│                         │       * F2: 分量重构 (非删除)     │   │
│                         └──────────────────────────────────┘   │
│                                    │                               │
│                                    ▼                               │
│                         ┌──────────────────────────────────┐   │
│                         │  语法检查 + 有效候选列表            │   │
│                         └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 四向变异算子定义（v2.0）

| 算子 | 名称 | 约束描述 | 适用场景 |
|------|------|----------|----------|
| **F1** | Branch Augmentation（分支扩充） | 完全保留原代码逻辑和权重，**仅新增一个**奖励或惩罚分量 | 诊断报告显示缺少某种协同行为引导 |
| **F2** | Component Reconstruction（分量重构） | 重构失效或起反作用分量的**数学表达形式**（如：线性→指数惩罚、引入平滑阈值、改变距离计算方式）。**保留分量数量不变，仅改变计算逻辑** | 诊断报告显示某分量持续为零或起反作用，但其物理维度有价值 |
| **F3** | Equilibrium Tuning（平衡微调） | **绝对不增加或删除**逻辑分支，严格保持代码拓扑不变，**仅修改权重系数** | 诊断报告显示整体结构合理但权重失衡 |
| **L1** | Paradigm Leap（范式跃迁） | 彻底抛弃原代码设计思路，从零构建**全新奖励函数**（如全局势场、极坐标等） | 诊断报告显示存在根本性问题，需要完全重构 |

### 2.3 关键设计原则

1. **自适应分配**：Reflection 阶段根据训练结果自主选择算子组合，而非静态循环分配
2. **基数约束**：每种算子最多分配 2 次，防止 LLM 输出格式错误时导致搜索空间塌陷
3. **F2 重构保护**：F2 算子从"破坏性删除"改为"保守性重构"，保留物理维度信息
4. **兜底机制**：若解析失败，自动回退到标准正交分配
5. **并发波束搜索**：通过 `ThreadPoolExecutor` 真正并发地向 LLM 发起独立请求

---

## 3. 详细实现

### 3.1 `prompt_templates.py` — 重构方法

#### `reflection_prompt`（v2.0 重构）

```139:171:llm_reward_agent/agent/prompt_templates.py
    @staticmethod
    def reflection_prompt(training_logs: str) -> str:
        """
        DREAM 模块：带自适应算子分配的诊断反思提示词
        强制 LLM 在输出诊断后，严格按规定格式输出下一代的算子分配方案。
        """
```

**核心变化**：

- 新增"算子分配"任务，要求 LLM 为 4 个候选独立分配算子
- 新增硬性约束："每种算子最多只能被选择 2 次"
- 新增强制输出格式：`[病理诊断]` 和 `[算子分配]` 两个区块

**输出格式约束**：

```
[病理诊断]
(你的诊断内容，限200字)

[算子分配]
Candidate 0: <F1/F2/F3/L1>
Candidate 1: <F1/F2/F3/L1>
Candidate 2: <F1/F2/F3/L1>
Candidate 3: <F1/F2/F3/L1>
```

#### `evoleap_prompt`（F2 算子重定义）

```433:439:llm_reward_agent/agent/prompt_templates.py
        # 定义四向变异策略
        strategies = {
            'F1': '【Reward Branch Augmentation (分支扩充)】\n请完全保留原代码的现有逻辑和权重，新增一个（且仅新增一个）奖励或惩罚分量，用于解决诊断报告中缺失的协同行为引导。',
            'F2': '【Reward Component Reconstruction (失效分量重构)】\n请定位诊断报告中指出的"失效"或"起反作用"的分量。不要直接删除它们，而是重构其数学逻辑（例如：将线性惩罚改为指数惩罚、引入平滑阈值或改变距离函数的计算方式）。保持其他有效分量不变。',
            'F3': '【Reward Equilibrium Tuning (平衡微调)】\n绝对不要增加或删除现有的逻辑分支！请严格保持代码拓扑不变，仅根据诊断报告，修改各奖励分量的权重系数（增大/减小）。',
            'L1': '【Reward Paradigm Leap (范式跃迁)】\n彻底抛弃原代码的设计思路！请从零开始构建一个全新的奖励函数（例如尝试全局势场、相对距离极坐标系等与原先完全不同的视角）。'
        }
```

**F2 变化对比**：

| 版本 | F2 定义 | 策略 |
|------|---------|------|
| v1.2 | Prune Refinement（修剪精炼） | 直接删除失效分量，不增加任何新逻辑 |
| v2.0 | Component Reconstruction（分量重构） | 重构失效分量的数学逻辑，保持分量数量不变 |

---

### 3.2 `reward_design_agent.py` — 自适应分配实现

#### `generate_candidates()` 进化代逻辑重构

```236:325:llm_reward_agent/agent/reward_design_agent.py
                    else:
                        # ==========================================
                        # 阶段三：基于 DREAM 算子执行自适应并行突变
                        # ==========================================
                        from collections import Counter
                        
                        # ... 获取父代数据 ...
                        
                        # ---------------------------------------------------
                        # 核心解析逻辑：从反思日志中提取算子分配并执行基数校验
                        # ---------------------------------------------------
                        assigned_operators = []
                        # 使用正则匹配 Candidate X: F1 格式
                        matches = re.findall(r'Candidate \d+:\s*(F1|F2|F3|L1)', reflection, re.IGNORECASE)

                        if len(matches) >= n_candidates:
                            assigned_operators = [m.upper() for m in matches[:n_candidates]]
                            # 基数约束校验：每种算子最多2次
                            counts = Counter(assigned_operators)
                            is_valid = all(v <= 2 for v in counts.values())

                            if is_valid:
                                print(f"   ✅ 成功解析自适应算子: {assigned_operators} (约束校验通过)")
                            else:
                                print(f"   ⚠️ LLM违反基数约束 {dict(counts)}，触发强制纠正。")
                                assigned_operators = []  # 触发兜底机制
                        else:
                            print(f"   ⚠️ 未能在反思中解析到规范的算子分配，触发兜底分配。")

                        # 兜底机制：恢复标准波束搜索
                        if not assigned_operators:
                            operators = ['F1', 'F2', 'F3', 'L1']
                            assigned_operators = [operators[i % 4] for i in range(n_candidates)]
                            print(f"   🔧 启用标准正交分配: {assigned_operators}")
                        # ---------------------------------------------------
```

**关键设计点**：

1. **正则解析器**：使用 `re.findall(r'Candidate \d+:\s*(F1|F2|F3|L1)', reflection)` 提取算子分配
2. **基数约束校验**：使用 `Counter` 统计每种算子出现次数，校验是否满足 ≤ 2 次约束
3. **异常处理分支**：
   - LLM 输出格式正确 + 约束满足 → 直接使用解析结果
   - LLM 输出格式正确 + 约束违反 → 触发兜底分配
   - LLM 输出格式错误（解析失败）→ 触发兜底分配
4. **兜底机制**：回退到标准正交分配 `['F1', 'F2', 'F3', 'L1']` 循环

---

## 4. 使用说明

### 4.1 正常执行流程

当 `generation > 0` 且存在父代数据时，控制台输出示例：

```
==================================================
🤖 第 1 代: 生成候选奖励函数
==================================================
🧬 执行 DREAM 自适应变异... (尝试 1/3)
   ✅ 成功解析自适应算子: ['F1', 'F2', 'F2', 'L1'] (约束校验通过)
   启动 4 个并发自适应变异线程...
   ✅ 候选 0: 语法检查通过
   ✅ 候选 1: 语法检查通过
   ✅ 候选 2: 语法检查通过
   ✅ 候选 3: 语法检查通过

✅ 成功生成 4 个有效候选
```

### 4.2 基数约束校验示例

```
🧬 执行 DREAM 自适应变异... (尝试 1/3)
   ⚠️ LLM违反基数约束 {'F3': 3, 'L1': 1}，触发强制纠正。
   🔧 启用标准正交分配: ['F1', 'F2', 'F3', 'L1']
   启动 4 个并发自适应变异线程...
```

### 4.3 解析失败兜底示例

```
🧬 执行 DREAM 自适应变异... (尝试 1/3)
   ⚠️ 未能在反思中解析到规范的算子分配，触发兜底分配。
   🔧 启用标准正交分配: ['F1', 'F2', 'F3', 'L1']
   启动 4 个并发自适应变异线程...
```

---

## 5. 与 CoT 两阶段管线的协作关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整进化流程                                     │
│                                                                      │
│  generation == 0:                                                    │
│    CoT阶段一（低T=0.3）→ CoT阶段二（高T）→ 训练 → DREAM诊断+分配       │
│                                                                      │
│  generation > 0:                                                      │
│    获取父代 + DREAM反思 → 自适应分配F1/F2/F3/L1 → 并发变异 → 训练     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            DREAM 对 Reflection 的升级                          │  │
│  │                                                              │  │
│  │  v1.2: Reflection 仅输出"病症"（分量贡献度/任务瓶颈/协同缺陷）    │  │
│  │  v2.0: Reflection 输出"病症" + "算子分配方案"（基数约束 ≤2）    │  │
│  │         LLM 根据诊断结果自主选择算子组合                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 算法伪代码（Algorithm: DREAM Operator）

```
Algorithm 1: DREAM Operator
Input: parent_code, reflection, n_candidates
Output: assigned_operators (list of operator types)

1  assigned_operators ← []
2  matches ← RegexFind(reflection, "Candidate \d+:\s*(F1|F2|F3|L1)")
3  
4  if len(matches) ≥ n_candidates then
5      assigned_operators ← [m.upper() for m in matches[:n_candidates]]
6      counts ← Counter(assigned_operators)
7      is_valid ← ∀v ∈ counts.values(): v ≤ 2
8      
9      if ¬is_valid then
10         Print "LLM violates cardinality constraint {counts}, using fallback"
11         assigned_operators ← []   // trigger fallback
12     else
13         Print "Successfully parsed adaptive operators: {assigned_operators}"
14 else
15     Print "Failed to parse operators from reflection, using fallback"
16
17 // Fallback: standard orthogonal allocation
18 if not assigned_operators then
19     operators ← ['F1', 'F2', 'F3', 'L1']
20     assigned_operators ← [operators[i mod 4] for i in range(n_candidates)]
21     Print "Fallback: {assigned_operators}"
22
23 return assigned_operators
```

---

## 7. 已知约束与限制

| 约束 | 说明 |
|------|------|
| 需要父代数据 | 若 `memory.get_best_code()` 或 `memory.get_reflection()` 失败，DREAM 无法执行，自动回退到 Zero-Shot |
| API 并发限制 | LLM 服务商可能对单用户并发数有限制；若频繁超时，可降低 `num_candidates` 或在配置中调整 `timeout` |
| LLM 输出格式依赖 | DREAM 依赖 LLM 输出规范的 `[算子分配]` 格式；若 LLM 持续输出错误格式，将持续触发兜底机制 |
| 基数约束 ≠ 多样性保证 | 每种算子最多 2 次不等于每代恰好使用 4 种算子；若 LLM 分配 `['F1', 'F1', 'F3', 'F3']`，约束满足但 L1 未被使用 |

---

## 8. 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `llm_reward_agent/agent/prompt_templates.py` | 重构方法 | 重构 `reflection_prompt` 为带自适应算子分配的诊断风格；重定义 F2 算子为"分量重构" |
| `llm_reward_agent/agent/reward_design_agent.py` | 新增导入 + 重写分支 | 新增 `Counter` 导入；重写 `generate_candidates` 中 `generation > 0` 的分支，加入正则解析和基数校验 |

---

## 9. 参考文献

- **EvoLeap v1.2**：EvoLeap 四向变异算子并行波束搜索（2026-04-07）
- **DREAM v2.0**：诊断反思驱动的自适应进化算子（2026-04-13）
