# 阶段二快速开始指南

## 环境准备

### 1. 安装依赖

```bash
# 安装阶段二所需的依赖包
pip install -r requirements_llm.txt
```

或者手动安装：

```bash
pip install openai>=1.0.0 pyyaml>=6.0 numpy>=1.24.0 matplotlib>=3.7.0
```

### 2. 配置API密钥

**方法1：环境变量（推荐）**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Windows CMD
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"
```

**方法2：配置文件**

编辑 `llm_reward_agent/config/llm_config.yaml`:

```yaml
llm:
  api_key: "your_api_key_here"  # 不推荐，有安全风险
```

## 快速测试

### 1. 基础功能测试（不需要API）

```bash
python quick_test_phase2.py
```

这个测试会验证：
- 模块导入
- 提示词生成
- 进化记忆管理
- 环境上下文提取

### 2. 完整功能测试（需要API密钥）

```bash
python test_phase2.py
```

## 使用示例

### 示例1：生成初始提示词

```python
from llm_reward_agent.agent import PromptTemplates

# 构造环境上下文
env_context = {
    'env_name': 'simple_tag_env',
    'observation_space': 'Box(16,)',
    'action_space': 'Box(2,)',
    'agent_info': {
        'num_adversaries': 3,
        'num_good': 1
    },
    'physical_constants': {
        'max_force': 1.0,
        'capture_threshold': 0.5
    }
}

# 生成提示词
prompt = PromptTemplates.initial_generation_prompt(
    env_context,
    task_description="3个追捕者围捕1个目标"
)

print(f"提示词长度: {len(prompt)} 字符")
```

### 示例2：使用进化记忆

```python
from llm_reward_agent.agent import EvolutionaryMemory

# 创建记忆实例
memory = EvolutionaryMemory(save_dir="experiments/evolution_archive")

# 保存一代记录
memory.save(
    generation=0,
    best_code="def compute_reward(...): return 0, {}",
    reflection="需要增加距离奖励",
    all_results=[
        {'id': 0, 'fitness': 0.8, 'status': 'success'},
        {'id': 1, 'fitness': 0.75, 'status': 'success'}
    ]
)

# 获取最优代码
best_code = memory.get_best_code(0)

# 导出摘要
summary = memory.export_summary()
print(summary)
```

### 示例3：完整Agent使用（需要API密钥）

```python
from llm_reward_agent.agent import RewardDesignAgent

# 初始化智能体
agent = RewardDesignAgent(
    config_path="llm_reward_agent/config/llm_config.yaml"
)

# 初始化任务
agent.initialize(
    env_file_path="MADDPG/envs/simple_tag_env.py",
    task_description="""
    任务：3个追捕者围捕1个目标
    要求：接近、包围、避免碰撞、快速完成
    """
)

# 运行一代进化（使用模拟训练）
result = agent.step(generation=0)

print(f"最优Fitness: {result['best_fitness']:.4f}")
print(f"反思: {result['reflection'][:200]}...")
```

## 配置说明

编辑 `llm_reward_agent/config/llm_config.yaml` 来调整参数：

```yaml
# LLM配置
llm:
  provider: "openai"          # LLM提供商
  model: "gpt-4"              # 模型名称
  temperature: 0.8            # 生成温度

# 生成配置
generation:
  num_candidates: 4           # 每代生成的候选数量
  temperature: 0.8            # 生成温度（0.7-1.0）
  max_tokens: 2500            # 最大Token数

# 进化配置
evolution:
  max_generations: 10         # 最大进化代数
  population_size: 4          # 种群大小

# Fitness权重
fitness:
  weights:
    success_rate: 1.0         # 成功率权重
    capture_time: -0.001      # 时间权重（负数）
    formation_quality: 0.3    # 队形质量权重
```

## 常见问题

### Q1: ModuleNotFoundError: No module named 'openai'

**解决**: 安装依赖包

```bash
pip install -r requirements_llm.txt
```

### Q2: API密钥未设置错误

**解决**: 设置环境变量或在配置文件中指定API密钥

```bash
$env:OPENAI_API_KEY="your_api_key"
```

### Q3: 中文字符显示乱码

**解决**: Windows PowerShell编码问题，这不影响功能，只是显示问题

### Q4: 如何使用DeepSeek等国产模型？

**解决**: 修改配置文件

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
```

## 下一步

- 查看完整文档：`PHASE2_DOCUMENTATION.md`
- 查看总体计划：`IMPLEMENTATION_PLAN.md`
- 运行完整测试：`python test_phase2.py`
- 开始阶段三开发：并行训练框架

## 技术支持

- 项目仓库：LEMS
- 问题反馈：提交 Issue
- 开发文档：PHASE2_DOCUMENTATION.md
