# 阶段四快速开始指南

**立即开始使用LEMS系统！**

---

## ⚡ 快速开始（5分钟）

### 1. 环境准备

```bash
# 激活conda环境
conda activate MPE

# 检查依赖（应该在阶段二已安装）
pip list | grep -E "openai|pyyaml|numpy|matplotlib"
```

### 2. 设置API密钥

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Windows CMD
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

### 3. 运行快速测试

```bash
# 快速测试（模拟训练，约5分钟）
python run_evolution.py --num_generations 3 --no-real-training
```

---

## 🚀 使用示例

### 示例1: 快速验证（推荐新手）

```bash
python run_evolution.py --num_generations 3 --no-real-training
```

- **耗时**: 约5分钟
- **成本**: $0.03（使用gpt-3.5-turbo更便宜）
- **用途**: 验证系统正常工作

### 示例2: 标准训练

```bash
python run_evolution.py --num_generations 5 --episode_num 100
```

- **耗时**: 约30分钟
- **成本**: $0.05（gpt-3.5）或 $2（gpt-4）
- **用途**: 获得可用的奖励函数

### 示例3: 完整进化

```bash
python run_evolution.py --num_generations 10 --episode_num 200 --max_workers 4
```

- **耗时**: 约2-3小时
- **成本**: $0.10（gpt-3.5）或 $4（gpt-4）
- **用途**: 论文实验数据

---

## 📊 查看结果

### 1. 查看摘要

```bash
# Windows
type experiments\evolution_run\evolution_summary.txt

# Linux/Mac
cat experiments/evolution_run/evolution_summary.txt
```

### 2. 生成可视化图表

```bash
python visualization/evolution_plot.py --archive_dir experiments/evolution_run/evolution_archive
```

生成的图表：
- `evolution_curve.png` - 进化曲线
- `fitness_distribution.png` - Fitness分布
- `success_rate.png` - 成功率对比
- `dashboard.png` - 综合仪表板

### 3. 使用最优代码

```bash
# 查看最优代码
type experiments\evolution_run\reward_function_best.py

# 复制到MADDPG目录
copy experiments\evolution_run\reward_function_best.py MADDPG\envs\reward_function.py

# 用最优代码重新训练
python MADDPG/main_train.py --episode_num 1000
```

---

## 🔧 参数调优

### 降低成本

```bash
# 使用gpt-3.5-turbo（修改配置文件）
# llm_config.yaml:
#   model: "gpt-3.5-turbo"

# 减少候选数量
#   num_candidates: 2

# 减少代数
python run_evolution.py --num_generations 3
```

### 提高质量

```bash
# 使用gpt-4
# llm_config.yaml:
#   model: "gpt-4"

# 增加候选数量
#   num_candidates: 6

# 增加训练轮数
python run_evolution.py --episode_num 200
```

### 加快速度

```bash
# 减少训练轮数
python run_evolution.py --episode_num 50

# 增加并行数（如果CPU核心足够）
python run_evolution.py --max_workers 8
```

---

## 🐛 常见问题

### Q1: ModuleNotFoundError: No module named 'openai'

**解决**: 安装依赖

```bash
pip install -r requirements_llm.txt
```

### Q2: API密钥错误

**解决**: 检查环境变量

```bash
# 检查是否设置
echo $env:OPENAI_API_KEY  # PowerShell
echo %OPENAI_API_KEY%     # CMD

# 重新设置
$env:OPENAI_API_KEY="your_key"
```

### Q3: 训练超时

**解决**: 增加超时时间或减少训练轮数

```bash
# 方法1: 增加超时（修改配置文件）
# llm_config.yaml:
#   training:
#     timeout: 2400  # 40分钟

# 方法2: 减少训练轮数
python run_evolution.py --episode_num 50
```

### Q4: 内存不足

**解决**: 减少并行数

```bash
python run_evolution.py --max_workers 2
```

### Q5: 想要使用DeepSeek等国产模型

**解决**: 修改配置文件

```yaml
# llm_config.yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
```

---

## 📈 进阶使用

### 1. 使用自定义任务描述

```bash
python run_evolution.py \
    --task_description "任务：4个追捕者围捕2个目标" \
    --num_generations 5
```

### 2. 自定义保存目录

```bash
python run_evolution.py \
    --save_dir my_experiments/run_001 \
    --num_generations 5
```

### 3. 自动复制最优代码

```bash
python run_evolution.py \
    --num_generations 5 \
    --copy_to_maddpg
```

这会将最优代码自动复制到 `MADDPG/envs/reward_function_evolved.py`

---

## 🎯 推荐工作流

### 新手流程

1. **快速验证**（5分钟）
   ```bash
   python run_evolution.py --num_generations 2 --no-real-training
   ```

2. **小规模测试**（15分钟）
   ```bash
   python run_evolution.py --num_generations 3 --episode_num 50
   ```

3. **查看结果**
   ```bash
   python visualization/evolution_plot.py
   ```

### 实验流程

1. **基准实验**
   - 手工设计3个奖励函数
   - 完整训练（5000回合）
   - 记录性能

2. **LLM实验**
   ```bash
   python run_evolution.py --num_generations 10 --episode_num 200
   ```

3. **对比分析**
   - 使用可视化工具生成图表
   - 统计分析
   - 撰写论文

---

## 📞 获取帮助

```bash
# 查看命令帮助
python run_evolution.py --help

# 查看可视化帮助
python visualization/evolution_plot.py --help
```

**文档资源**:
- `PHASE4_DOCUMENTATION.md` - 完整开发文档
- `IMPLEMENTATION_PLAN.md` - 总体计划
- `README_PHASE123.md` - 项目总览

**测试脚本**:
- `quick_test_phase4.py` - 快速功能验证
- `test_phase4.py` - 完整测试

---

## ✅ 验收清单

运行进化前请确认：

- [ ] Python 3.11.8环境已激活
- [ ] 依赖包已安装（openai, pyyaml, numpy, matplotlib）
- [ ] API密钥已设置（OPENAI_API_KEY）
- [ ] MADDPG代码完整（envs/, agents/, utils/等）
- [ ] 磁盘空间充足（建议>1GB）

---

**准备好了？现在就开始你的第一次进化吧！** 🚀

```bash
python run_evolution.py --num_generations 3 --no-real-training
```
