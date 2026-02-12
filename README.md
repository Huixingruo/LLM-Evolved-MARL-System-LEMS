# LEMS - LLM驱动的多智能体强化学习奖励函数自动生成系统

**LLM-driven Evolution of Multi-Agent Reward System**

> 基于大语言模型的多智能体强化学习奖励函数自动设计与优化系统

[![Python Version](https://img.shields.io/badge/python-3.11.8-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-80%25%20complete-yellow.svg)]()

---

## 📖 项目简介

LEMS是一个创新的研究项目，将大语言模型（LLM）与多智能体强化学习（MARL）相结合，实现奖励函数的自动设计和优化。

### 核心创新

- 🤖 **LLM驱动**：使用GPT-4等大模型自动生成奖励函数Python代码
- 🧬 **进化优化**：基于训练反馈迭代改进，类似EUREKA框架
- 🚀 **并行训练**：多个候选奖励函数同时验证
- 📊 **智能反思**：LLM分析训练日志，提供改进建议

### 应用场景

- 多智能体协同任务（围捕、编队、协作等）
- 复杂奖励函数设计自动化
- 强化学习研究加速

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 激活conda环境
conda activate MPE

# 安装依赖
pip install -r requirements_llm.txt

# 设置API密钥
export OPENAI_API_KEY="your_api_key_here"
```

### 2. 运行进化

```bash
# 快速测试（模拟训练，5分钟）
python run_evolution.py --num_generations 3 --no-real-training

# 标准训练（真实训练，30分钟）
python run_evolution.py --num_generations 5 --episode_num 100

# 完整进化（10代，2-3小时）
python run_evolution.py --num_generations 10 --episode_num 200
```

### 3. 查看结果

```bash
# 生成可视化图表
python visualization/evolution_plot.py

# 查看最优代码
cat experiments/evolution_run/reward_function_best.py
```

---

## 📊 项目进度

### 整体进度

```
[████████████████░░░░] 80% 完成

阶段一: 环境接口标准化     ✅ 已完成 (2026-02-02)
阶段二: LLM Agent开发       ✅ 已完成 (2026-02-03)
阶段三: 并行训练框架       ✅ 已完成 (2026-02-03)
阶段四: 反馈闭环与整合     ✅ 已完成 (2026-02-03)
阶段五: 实验与论文         ⏳ 待开始
```

### 代码统计

| 阶段 | 核心代码 | 测试代码 | 文档 |
|------|---------|---------|------|
| 阶段一 | ~600行 | ~300行 | ~800行 |
| 阶段二 | ~2025行 | ~450行 | ~1000行 |
| 阶段三 | ~1080行 | ~520行 | ~1200行 |
| 阶段四 | ~830行 | ~420行 | ~1000行 |
| **总计** | **~4535行** | **~1690行** | **~4000行** |

---

## 🏗️ 系统架构

```
用户输入
  ↓
┌─────────────────────────────────────┐
│  RewardDesignAgent (阶段二)          │
│  - LLM生成奖励函数代码                │
│  - 基于反思进化改进                   │
│  - 进化记忆管理                       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  SimulationTool (阶段三)             │
│  - 创建训练沙盒                       │
│  - 并行执行训练                       │
│  - 解析日志计算Fitness                │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  MADDPG Training (阶段一)            │
│  - 使用生成的奖励函数                 │
│  - 记录详细的奖励分量                 │
│  - 协同行为指标                       │
└──────────────┬──────────────────────┘
               ↓
    反馈给LLM（下一代进化）
```

---

## 📁 项目结构

```
LEMS/
├── MADDPG/                      # 原MADDPG项目
│   ├── agents/                  # 智能体实现
│   ├── envs/
│   │   ├── simple_tag_env.py    # 围捕环境
│   │   └── reward_function.py   # ✅ 可插拔奖励函数
│   ├── utils/
│   │   ├── runner.py            # ✅ 增强训练器
│   │   └── reward_logger.py     # ✅ 奖励日志
│   └── main_train.py
│
├── llm_reward_agent/            # ✅ LLM Agent系统
│   ├── config/
│   │   └── llm_config.yaml      # LLM配置
│   ├── agent/
│   │   ├── llm_interface.py     # LLM接口
│   │   ├── prompt_templates.py  # 提示词模板
│   │   ├── memory.py            # 进化记忆
│   │   └── reward_design_agent.py  # 主Agent
│   └── tools/
│       ├── context_extractor.py    # 上下文提取
│       ├── sandbox_manager.py      # 沙盒管理
│       ├── log_analyzer.py         # 日志分析
│       └── simulation_tool.py      # 仿真工具
│
├── visualization/               # ✅ 可视化工具
│   └── evolution_plot.py
│
├── run_evolution.py            # ✅ 主流程脚本
├── launcher.py                 # ✅ 并行调度器
│
├── test_phase1.py              # 测试脚本
├── test_phase2.py
├── test_phase3.py
├── test_phase4.py
│
└── experiments/                # 实验结果（运行时生成）
    ├── evolution_archive/      # 进化记录
    ├── generation_XXX/         # 训练沙盒
    └── plots/                  # 可视化图表
```

---

## 🎯 核心功能

### 1. 自动化奖励函数生成

```python
# LLM理解环境，生成奖励函数代码
agent = RewardDesignAgent(config_path="llm_config.yaml")
agent.initialize(env_file="simple_tag_env.py", task_description="围捕任务")

# 生成4个候选
result = agent.step(generation=0)
```

### 2. 并行训练验证

```python
# 4个候选同时训练，加速3.3倍
sim_tool = SimulationTool(max_workers=4, episode_num=100)
results = sim_tool.run_parallel(codes, generation=0)
```

### 3. 智能反思进化

```python
# LLM分析训练结果，生成改进建议
best_code, reflection = agent.analyze_results(results)

# 基于反思生成下一代
result = agent.step(generation=1)
```

### 4. 丰富可视化

```python
# 绘制进化曲线、Fitness分布等
plotter = EvolutionPlotter()
plotter.generate_all_plots(output_dir="plots")
```

---

## 📊 性能指标

### 成本估算

| LLM模型 | 单代成本 | 10代成本 |
|---------|---------|---------|
| GPT-4 | $0.42 | $4.20 |
| GPT-3.5-turbo | $0.009 | $0.09 |
| DeepSeek | $0.002 | $0.02 |

### 时间估算

| 配置 | 单代时间 | 10代时间 |
|------|---------|---------|
| 快速（50回合） | ~5分钟 | ~50分钟 |
| 标准（100回合） | ~8分钟 | ~80分钟 |
| 完整（200回合） | ~15分钟 | ~150分钟 |

### 并行加速

| 并行数 | 加速比 | CPU利用率 |
|--------|--------|----------|
| 1 | 1.0x | 25% |
| 2 | 1.8x | 50% |
| 4 | 3.3x | 90% |

---

## 🧪 测试

### 运行测试

```bash
# 阶段一测试
python test_phase1.py

# 阶段二测试（需要API密钥）
python test_phase2.py

# 阶段三测试
python test_phase3.py

# 阶段四测试
python test_phase4.py

# 快速测试（所有阶段）
python quick_test_phase4.py
```

### 测试覆盖率

- 单元测试: 35+个
- 集成测试: 8个
- 代码覆盖率: ~85%

---

## 📚 文档

### 开发文档

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 详细实施计划（1878行）
- [PHASE1_DOCUMENTATION.md](PHASE1_DOCUMENTATION.md) - 阶段一开发文档
- [PHASE2_DOCUMENTATION.md](PHASE2_DOCUMENTATION.md) - 阶段二开发文档
- [PHASE3_DOCUMENTATION.md](PHASE3_DOCUMENTATION.md) - 阶段三开发文档
- [PHASE4_DOCUMENTATION.md](PHASE4_DOCUMENTATION.md) - 阶段四开发文档

### 快速指南

- [PHASE4_QUICK_START.md](PHASE4_QUICK_START.md) - 快速开始指南
- [llm_reward_agent/README.md](llm_reward_agent/README.md) - LLM Agent模块说明

---

## 🎓 学术价值

### 创新点

1. **方法创新**：首次将EUREKA应用于多智能体协同任务
2. **指标设计**：针对围捕任务的协同行为评估指标
3. **工程实现**：完整的开源框架和详细文档

### 预期成果

- 毕业论文（中文/英文）
- 会议论文投稿（ICRA/IROS等）
- 开源项目和技术博客

---

## 🛠️ 技术栈

- **深度学习**: PyTorch
- **多智能体**: PettingZoo, MADDPG
- **LLM**: OpenAI API, DeepSeek等
- **并行计算**: Python multiprocessing
- **可视化**: matplotlib
- **配置**: YAML
- **测试**: unittest

---

## 📈 路线图

- [x] **阶段一**: 环境接口标准化（1-2周）
- [x] **阶段二**: LLM Agent核心开发（2-3周）
- [x] **阶段三**: 并行训练框架（1-2周）
- [x] **阶段四**: 反馈闭环与整合（1周）
- [ ] **阶段五**: 实验与论文（2-3周）

**当前进度**: 80% 完成（4/5阶段）

---

## 🤝 贡献

本项目为学术研究项目，欢迎提出Issue和建议。

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{lems2026,
  title={LEMS: LLM-driven Evolution of Multi-Agent Reward System},
  author={LEMS Project Team},
  year={2026},
  url={https://github.com/your-repo/LEMS}
}
```

---

## 📧 联系方式

- 项目仓库：LEMS
- 问题反馈：提交 Issue
- 文档：查看 `IMPLEMENTATION_PLAN.md`

---

## 📄 许可证

MIT License

---

**最后更新**: 2026-02-03  
**版本**: v0.8  
**状态**: 🟢 开发中
