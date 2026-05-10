# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LEMS (LLM-driven Evolution of Multi-Agent Reward System) 是一个基于大语言模型的多智能体强化学习奖励函数自动设计与优化系统。系统使用GPT-4等大模型自动生成奖励函数Python代码，并通过进化优化迭代改进。

## 核心命令

### 环境配置
```bash
conda activate MPE
pip install -r MADDPG/utils/pip-requirements.txt
export OPENAI_API_KEY="your_api_key_here"
```

### 运行进化
```bash
# 快速测试（模拟训练，5分钟）
python run_evolution.py --num_generations 3 --no-real-training

# 标准训练（真实训练，30分钟）
python run_evolution.py --num_generations 5 --episode_num 100

# 完整进化（10代，2-3小时）
python run_evolution.py --num_generations 10 --episode_num 200

# 从中断处继续训练
python run_evolution.py --resume --archive_dir experiments/evolution_archive --num_generations 5
```

### 测试
```bash
# 运行所有测试
python tests/run_all_tests.py

# 运行特定阶段测试
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py

# 快速测试
python quick_test_phase4.py
```

### 可视化
```bash
python visualization/evolution_plot.py
```

## 架构概览

### 系统流程
```
用户输入任务描述 → LLM生成奖励函数代码 → 并行训练验证 → LLM分析训练结果 → 迭代进化 → 输出最优奖励函数
```

### 核心模块

**环境层 (MADDPG/)**
- `envs/reward_function.py` - 可插拔奖励函数接口，LLM生成并替换`compute_reward`函数
- `utils/runner.py` - 增强训练器
- `utils/reward_logger.py` - 奖励分量日志记录器
- `main_train.py` - MADDPG训练入口

**Agent层 (llm_reward_agent/)**
- `agent/reward_design_agent.py` - 主Agent类，协调完整进化流程
- `agent/llm_interface.py` - 统一LLM接口（支持OpenAI、DeepSeek等）
- `agent/prompt_templates.py` - 提示词模板（生成、反思、改进）
- `agent/memory.py` - 进化记忆管理（持久化+可视化）

**工具层 (llm_reward_agent/tools/)**
- `context_extractor.py` - 环境上下文自动提取
- `sandbox_manager.py` - 沙盒管理器（创建独立训练环境）
- `simulation_tool.py` - 仿真工具集成（统一并行训练接口）
- `log_analyzer.py` - 日志分析器（Fitness计算）

**并行调度 (launcher.py)**
- 支持CPU和GPU两种并行模式
- 使用multiprocessing.Pool或subprocess并行训练

**可视化 (visualization/)**
- `evolution_plot.py` - 进化曲线、Fitness分布等图表生成

### 关键配置文件

- `llm_reward_agent/config/llm_config.yaml` - LLM配置（模型、温度、token数等）
- 训练配置（回合数、并行数、超时时间等）
- Fitness计算权重配置

## 开发注意事项

### 代码结构
- 奖励函数必须实现`compute_reward(agent_name, observation, global_state, actions, world)`接口
- 返回值格式：`(reward, components_dict)`
- components_dict用于记录各奖励分量，便于日志分析

### 环境变量
- `OPENAI_API_KEY` - OpenAI API密钥（必需）
- `MPLBACKEND=Agg` - matplotlib非交互式后端（已自动设置）
- `KMP_DUPLICATE_LIB_OK=True` - 允许OpenMP库重复加载（已自动设置）

### 测试结构
- 测试文件位于`tests/`目录
- `tests/agent/` - Agent模块测试
- `tests/tools/` - 工具模块测试
- 使用`run_all_tests.py`运行完整测试套件

### 实验结果
- 实验结果保存在`experiments/`目录
- 每代训练结果在`experiments/generation_XXX/`目录
- 进化记录在`experiments/evolution_archive/`目录

## 技术栈

- **深度学习**: PyTorch
- **多智能体**: PettingZoo, MADDPG
- **LLM**: OpenAI API, DeepSeek等
- **并行计算**: Python multiprocessing
- **可视化**: matplotlib
- **配置**: YAML
- **测试**: unittest
