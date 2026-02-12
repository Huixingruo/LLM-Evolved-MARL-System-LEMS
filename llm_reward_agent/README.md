# LLM奖励函数设计智能体

> 基于大语言模型的多智能体强化学习奖励函数自动生成系统

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai pyyaml numpy matplotlib
```

### 2. 配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your_api_key_here"
```

### 3. 运行示例

```python
from llm_reward_agent.agent import RewardDesignAgent

# 初始化智能体
agent = RewardDesignAgent(config_path="llm_reward_agent/config/llm_config.yaml")

# 初始化任务
agent.initialize(
    env_file_path="MADDPG/envs/simple_tag_env.py",
    task_description="3个追捕者围捕1个目标"
)

# 运行一代进化
result = agent.step(generation=0)
print(f"Fitness: {result['best_fitness']:.4f}")
```

## 📁 项目结构

```
llm_reward_agent/
├── config/
│   └── llm_config.yaml         # LLM配置文件
├── agent/
│   ├── llm_interface.py         # LLM接口封装
│   ├── prompt_templates.py      # 提示词模板
│   ├── reward_design_agent.py   # 主Agent类
│   └── memory.py                # 进化记忆管理
└── tools/
    └── context_extractor.py     # 环境上下文提取
```

## 🔧 配置说明

编辑 `config/llm_config.yaml` 来调整参数：

```yaml
llm:
  model: "gpt-4"              # LLM模型
  temperature: 0.8            # 生成温度

generation:
  num_candidates: 4           # 每代候选数量
  temperature: 0.8

evolution:
  max_generations: 10         # 最大进化代数
```

## 📊 运行测试

```bash
python test_phase2.py
```

## 📖 详细文档

请参考：
- `PHASE2_DOCUMENTATION.md` - 完整开发文档
- `IMPLEMENTATION_PLAN.md` - 总体实施计划

## ⚡ 支持的LLM

- ✅ OpenAI (gpt-4, gpt-3.5-turbo)
- ✅ DeepSeek (deepseek-chat)
- ⚠️ Anthropic (待测试)
- ⚠️ 智谱AI (待测试)

## 📝 开发状态

- ✅ 阶段一：环境接口标准化
- ✅ 阶段二：LLM Agent核心开发
- ⏳ 阶段三：并行训练框架（开发中）
- ⏳ 阶段四：反馈闭环与整合
- ⏳ 阶段五：实验与论文

## 📧 联系方式

- 项目仓库：LEMS
- 问题反馈：请提交 Issue
