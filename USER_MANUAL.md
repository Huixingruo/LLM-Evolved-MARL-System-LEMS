# LEMS系统使用手册

**LLM驱动的多智能体强化学习奖励函数自动生成系统 - 完整使用指南**

> **版本**: v1.0  
> **更新日期**: 2026-02-03  
> **适用版本**: LEMS v0.8+

---

## 📑 目录

1. [系统简介](#1-系统简介)
2. [安装指南](#2-安装指南)
3. [快速开始](#3-快速开始)
4. [详细使用](#4-详细使用)
5. [高级功能](#5-高级功能)
6. [故障排除](#6-故障排除)
7. [常见问题](#7-常见问题)

---

## 1. 系统简介

### 1.1 LEMS是什么？

LEMS（LLM-driven Evolution of Multi-Agent Reward System）是一个自动化的奖励函数设计系统，它使用大语言模型（如GPT-4）为多智能体强化学习任务自动生成和优化奖励函数。

### 1.2 核心功能

- 🤖 **自动生成**: LLM理解任务，生成Python代码
- 🧬 **智能进化**: 基于训练反馈迭代优化
- 🚀 **并行训练**: 多个候选同时验证
- 📊 **专业可视化**: 生成论文级别的图表

### 1.3 适用场景

- 多智能体协同任务（围捕、编队等）
- 需要快速设计奖励函数
- 奖励工程研究
- 强化学习教学

---

## 2. 安装指南

### 2.1 系统要求

**硬件要求**:
- CPU: 4核以上（推荐）
- 内存: 4GB以上
- 磁盘: 2GB以上

**软件要求**:
- Python: 3.11.8（推荐，已测试）
- 操作系统: Windows 10/11, Linux, macOS

### 2.2 安装步骤

#### 步骤1: 克隆项目

```bash
cd /path/to/your/workspace
# 如果是从git克隆
git clone <your-repo-url> LEMS
cd LEMS
```

#### 步骤2: 创建conda环境（推荐）

```bash
conda create -n lems python=3.11.8
conda activate lems
```

#### 步骤3: 安装依赖

```bash
# 安装MADDPG依赖（阶段一）
pip install torch pettingzoo numpy

# 安装LLM依赖（阶段二三四）
pip install -r requirements_llm.txt
```

#### 步骤4: 配置API密钥

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

**提示**: 建议将API密钥写入环境配置文件：
- Windows: `~/.bash_profile` 或 系统环境变量
- Linux/Mac: `~/.bashrc` 或 `~/.zshrc`

#### 步骤5: 验证安装

```bash
python quick_test_phase4.py
```

预期看到: "[OK] 阶段四开发完成！"

---

## 3. 快速开始

### 3.1 第一次运行（5分钟）

```bash
# 使用模拟训练，快速验证系统
python run_evolution.py --num_generations 3 --no-real-training
```

**预期结果**:
- 生成3代进化记录
- 保存最优奖励函数
- 约5分钟完成

### 3.2 查看结果

```bash
# 查看摘要
type experiments\evolution_run\evolution_summary.txt  # Windows
cat experiments/evolution_run/evolution_summary.txt   # Linux/Mac

# 查看最优代码
type experiments\evolution_run\reward_function_best.py
```

### 3.3 生成可视化

```bash
python visualization/evolution_plot.py
```

生成的图表在 `experiments/plots/` 目录。

---

## 4. 详细使用

### 4.1 运行模式

#### 模式1: 模拟训练（快速测试）

```bash
python run_evolution.py --num_generations 3 --no-real-training
```

**特点**:
- ✅ 快速（约5分钟）
- ✅ 不需要长时间训练
- ❌ 使用模拟数据（Fitness不真实）

**适用于**: 验证系统、调试、演示

#### 模式2: 真实训练（标准）

```bash
python run_evolution.py --num_generations 5 --episode_num 100
```

**特点**:
- ✅ 真实训练验证
- ✅ Fitness可信
- ⏱️ 耗时约30分钟

**适用于**: 日常使用、获得可用代码

#### 模式3: 完整进化（论文级）

```bash
python run_evolution.py --num_generations 10 --episode_num 200 --max_workers 4
```

**特点**:
- ✅ 充分优化
- ✅ 论文级别数据
- ⏱️ 耗时2-3小时

**适用于**: 论文实验、最终交付

### 4.2 参数说明

#### 基础参数

```bash
--config <path>              # LLM配置文件（默认: llm_config.yaml）
--num_generations <int>      # 进化代数（默认: 5）
--use_real_training          # 使用真实训练（默认开启）
--no-real-training           # 使用模拟训练
```

#### 训练参数

```bash
--episode_num <int>          # 每个候选训练回合数（默认: 100）
--max_workers <int>          # 并行进程数（默认: 4）
```

#### 环境参数

```bash
--env_file <path>            # 环境文件路径
--task_description <str>     # 任务描述
```

#### 保存参数

```bash
--save_dir <path>            # 结果保存目录
--copy_to_maddpg             # 复制最优代码到MADDPG目录
```

### 4.3 配置文件

编辑 `llm_reward_agent/config/llm_config.yaml`:

```yaml
# LLM配置
llm:
  provider: "openai"          # 或 "deepseek"
  model: "gpt-4"              # 或 "gpt-3.5-turbo", "deepseek-chat"
  api_key: "${OPENAI_API_KEY}"

# 生成配置
generation:
  num_candidates: 4           # 每代生成数量
  temperature: 0.8            # 生成温度（0.7-1.0）

# 训练配置
training:
  episode_num: 100            # 训练回合数
  parallel_workers: 4         # 并行数
  timeout: 1200               # 超时时间（秒）

# Fitness权重
fitness:
  weights:
    success_rate: 1.0         # 成功率权重
    capture_time: -0.001      # 时间权重（负数）
    formation_quality: 0.3    # 队形质量权重
```

---

## 5. 高级功能

### 5.1 使用不同LLM

#### GPT-3.5-turbo（便宜）

```yaml
# llm_config.yaml
llm:
  model: "gpt-3.5-turbo"
```

**成本**: 10代约$0.09

#### GPT-4（高质量）

```yaml
llm:
  model: "gpt-4"
```

**成本**: 10代约$4.20

#### DeepSeek（国产）

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
```

**成本**: 10代约$0.02

### 5.2 自定义任务

```python
# 创建自定义脚本
from llm_reward_agent.agent import RewardDesignAgent

agent = RewardDesignAgent(config_path="llm_config.yaml")

# 自定义任务描述
agent.initialize(
    env_file_path="your_env.py",
    task_description="""
    你的任务描述：
    - 目标1
    - 目标2
    - 约束条件
    """
)

# 运行进化
for gen in range(10):
    result = agent.step(gen, use_real_training=True)
```

### 5.3 批量实验

```bash
# 脚本: run_experiments.sh

# 实验1: 少代数
python run_evolution.py --num_generations 3 --save_dir exp1

# 实验2: 多代数
python run_evolution.py --num_generations 10 --save_dir exp2

# 实验3: 不同LLM（修改配置后）
python run_evolution.py --num_generations 5 --save_dir exp3
```

### 5.4 断点续训（手动）

```python
# 加载已有记忆
from llm_reward_agent.agent import EvolutionaryMemory

memory = EvolutionaryMemory(save_dir="experiments/evolution_archive")
memory.load_from_disk()

# 从第N代继续
start_gen = len(memory.history)

agent = RewardDesignAgent()
# ... 初始化 ...
agent.memory = memory  # 使用加载的记忆

for gen in range(start_gen, 10):
    result = agent.step(gen)
```

---

## 6. 故障排除

### 6.1 常见错误

#### 错误1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'openai'
```

**解决**:
```bash
pip install -r requirements_llm.txt
```

#### 错误2: API密钥错误

```
ValueError: API密钥未设置
```

**解决**:
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY="sk-..."
```

#### 错误3: 训练超时

```
[candidate_0] 训练超时
```

**解决**:
```yaml
# 修改 llm_config.yaml
training:
  timeout: 2400  # 增加到40分钟
```

或减少训练轮数:
```bash
python run_evolution.py --episode_num 50
```

#### 错误4: 内存不足

```
MemoryError: ...
```

**解决**:
```bash
# 减少并行数
python run_evolution.py --max_workers 2
```

### 6.2 调试技巧

#### 使用串行模式

```python
# 在simulation_tool.py中
results = sim_tool.run_sequential(codes, generation)  # 而不是run_parallel
```

#### 查看详细日志

```bash
# 查看沙盒中的训练输出
cat experiments/generation_000/candidate_0/MADDPG/logs/training_log.json
```

#### 检查生成的代码

```bash
# 查看LLM生成的代码
cat experiments/generation_000_summary.txt
```

---

## 7. 常见问题

### Q1: 如何降低成本？

**A**: 
1. 使用gpt-3.5-turbo（成本降低95%）
2. 使用DeepSeek等国产模型（成本更低）
3. 减少代数和候选数量

```yaml
# llm_config.yaml
llm:
  model: "gpt-3.5-turbo"

generation:
  num_candidates: 2  # 从4减少到2
```

### Q2: 如何加快速度？

**A**:
1. 减少训练回合数
2. 增加并行数（如果CPU允许）
3. 使用模拟模式测试

```bash
# 快速模式
python run_evolution.py --episode_num 50 --max_workers 8
```

### Q3: 生成的代码质量不好怎么办？

**A**:
1. 使用更好的模型（gpt-4）
2. 优化提示词
3. 增加生成的候选数量
4. 多运行几次，选择最好的

### Q4: 如何用于自己的环境？

**A**:
1. 确保环境实现了 `reward_function.py` 接口
2. 更新环境上下文提取器
3. 调整任务描述
4. 运行进化

### Q5: 训练失败怎么办？

**A**:
系统有自动容错：
- 重试3次
- 使用后备代码
- 跳过失败候选

你也可以：
- 检查环境配置
- 减少训练轮数
- 查看错误日志

### Q6: 如何对比多个LLM？

**A**:
```bash
# 运行GPT-4实验
# 修改配置: model: "gpt-4"
python run_evolution.py --num_generations 5 --save_dir exp_gpt4

# 运行GPT-3.5实验
# 修改配置: model: "gpt-3.5-turbo"
python run_evolution.py --num_generations 5 --save_dir exp_gpt35

# 运行DeepSeek实验
# 修改配置: provider: "deepseek", model: "deepseek-chat"
python run_evolution.py --num_generations 5 --save_dir exp_deepseek

# 对比结果
python compare_experiments.py exp_gpt4 exp_gpt35 exp_deepseek
```

### Q7: 生成的代码如何使用？

**A**:
```bash
# 方法1: 自动复制
python run_evolution.py --copy_to_maddpg

# 方法2: 手动复制
cp experiments/evolution_run/reward_function_best.py MADDPG/envs/reward_function.py

# 方法3: 在代码中使用
# 直接读取并替换reward_function.py
```

### Q8: 如何评估生成的奖励函数？

**A**:
```bash
# 使用评估脚本
python MADDPG/main_evaluate.py

# 查看训练曲线
python MADDPG/plot/plot_rewards.py

# 可视化行为
python MADDPG/main_evaluate_matplotlib.py
```

---

## 📖 推荐阅读顺序

### 新手

1. `README.md` - 项目总览
2. `PHASE4_QUICK_START.md` - 快速开始
3. 运行 `python demo_evolution.py` - 功能演示
4. 运行第一次进化

### 开发者

1. `IMPLEMENTATION_PLAN.md` - 总体设计
2. `PHASE2_DOCUMENTATION.md` - LLM Agent实现
3. `PHASE3_DOCUMENTATION.md` - 并行训练实现
4. `PHASE4_DOCUMENTATION.md` - 整合实现
5. 阅读源代码

### 研究者

1. `PROJECT_STATUS.md` - 项目状态
2. `PHASE234_COMPLETE_REPORT.md` - 完成报告
3. 运行完整实验
4. 分析结果，撰写论文

---

## 🎯 使用建议

### 建议的使用流程

1. **熟悉系统**（1天）
   - 运行演示
   - 阅读文档
   - 运行快速测试

2. **小规模实验**（1天）
   - 3代进化，模拟训练
   - 3代进化，真实训练（50回合）
   - 分析结果

3. **标准实验**（1天）
   - 5代进化，100回合
   - 生成可视化
   - 评估质量

4. **论文实验**（3-5天）
   - 10代进化，200回合
   - 对比不同LLM
   - 消融实验
   - 收集数据

### 成本优化建议

| 需求 | 推荐配置 | 成本 | 时间 |
|------|---------|------|------|
| 快速验证 | gpt-3.5, 3代, 模拟 | $0.03 | 5分钟 |
| 日常使用 | gpt-3.5, 5代, 100回合 | $0.05 | 30分钟 |
| 论文实验 | gpt-4, 10代, 200回合 | $4.20 | 3小时 |
| 成本敏感 | DeepSeek, 10代, 200回合 | $0.02 | 3小时 |

---

## 📞 技术支持

### 获取帮助

```bash
# 查看命令帮助
python run_evolution.py --help
python visualization/evolution_plot.py --help

# 运行测试诊断
python quick_test_phase4.py
python test_phase4.py
```

### 文档资源

- **快速开始**: `PHASE4_QUICK_START.md`
- **完整文档**: `PHASE4_DOCUMENTATION.md`
- **项目状态**: `PROJECT_STATUS.md`
- **完成报告**: `PHASE234_COMPLETE_REPORT.md`

### 问题反馈

- 提交Issue到项目仓库
- 附带错误日志和配置文件
- 描述复现步骤

---

## 📝 附录

### A. 完整命令参考

```bash
# run_evolution.py 所有参数
python run_evolution.py \
    --config llm_reward_agent/config/llm_config.yaml \
    --num_generations 5 \
    --use_real_training \
    --episode_num 100 \
    --max_workers 4 \
    --env_file MADDPG/envs/simple_tag_env.py \
    --task_description "3个追捕者围捕1个目标" \
    --save_dir experiments/my_run \
    --copy_to_maddpg
```

### B. 文件路径索引

**核心代码**:
- Agent: `llm_reward_agent/agent/`
- 工具: `llm_reward_agent/tools/`
- 配置: `llm_reward_agent/config/`

**脚本**:
- 主流程: `run_evolution.py`
- 并行器: `launcher.py`
- 可视化: `visualization/evolution_plot.py`

**测试**:
- 快速: `quick_test_phase4.py`
- 完整: `test_phase4.py`
- 演示: `demo_evolution.py`

**文档**:
- 总览: `README.md`
- 计划: `IMPLEMENTATION_PLAN.md`
- 各阶段: `PHASE[1-4]_DOCUMENTATION.md`

---

**使用手册版本**: v1.0  
**最后更新**: 2026-02-03  
**维护者**: LEMS Project Team

---

**祝你使用愉快！** 🎉
