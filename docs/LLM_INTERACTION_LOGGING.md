# LLM交互记录保存功能文档

**版本**: 1.0
**日期**: 2026-04-08
**作者**: LEMS Project

---

## 一、功能概述

LLM交互记录保存功能用于追踪和记录奖励函数设计智能体（RewardDesignAgent）与大语言模型（LLM）之间的每一次交互对话。该功能通过在关键阶段埋点，将所有发送给LLM的提示词（Prompt）和LLM返回的响应（Response）以Markdown格式保存到独立文件夹中，便于后期学术复现、日志分析和调试。

## 二、目录结构

保存路径遵循统一的目录组织规范：

```
save_dir/
└── llm_interactions/
    └── generation_X/
        ├── YYYYMMDD_HHMMSS_Phase1_CoT_Analysis.md
        ├── YYYYMMDD_HHMMSS_Phase2_Initial_Generation_candidate_0.md
        ├── YYYYMMDD_HHMMSS_Phase2_Initial_Generation_candidate_1.md
        ├── YYYYMMDD_HHMMSS_Phase3_EvoLeap_candidate_worker0_F1.md
        ├── YYYYMMDD_HHMMSS_Phase3_EvoLeap_candidate_worker1_F2.md
        └── YYYYMMDD_HHMMSS_Phase4_Reflection_candidate_best_code_diagnosis.md
```

- `save_dir`: 配置中指定的日志根目录（`config['logging']['save_dir']`）
- `llm_interactions`: 固定的交互记录子目录
- `generation_X`: 按代数组织的子目录
- 文件名格式: `时间戳_阶段_候选标识.md`

## 三、记录的阶段

功能覆盖了进化流程的四个关键阶段：

| 阶段 | 阶段标识 | 说明 |
|------|----------|------|
| 阶段一 | `Phase1_CoT_Analysis` | CoT思维链环境解析，生成MDP表征先验 |
| 阶段二 | `Phase2_Initial_Generation` | 基于CoT先验并行生成初始候选代码 |
| 阶段三 | `Phase3_EvoLeap` | 基于EvoLeap算子（F1/F2/F3/L1）的定向变异 |
| 阶段四 | `Phase4_Reflection` | 客观病理反思，生成诊断报告 |

## 四、文件格式

每个交互记录文件采用Markdown格式，包含以下内容：

```markdown
# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase1_CoT_Analysis
- **Candidate Info**: test_001
- **Timestamp**: 20260408_163910

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
[发送给LLM的完整提示词内容]
```

================================================================================
## Response (From LLM)
================================================================================

```text
[LLM返回的完整响应内容]
```
```

## 五、实现细节

### 5.1 核心方法

`_save_llm_interaction` 方法是整个功能的核心，负责构建目录结构、生成文件名和写入文件内容：

```python
def _save_llm_interaction(self, generation: int, phase: str,
                          prompt: str, response: str,
                          candidate_id: str = None):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = os.path.join(
        self.config['logging']['save_dir'],
        'llm_interactions',
        f'generation_{generation}'
    )
    os.makedirs(log_dir, exist_ok=True)

    filename_parts = [timestamp, phase]
    if candidate_id is not None:
        filename_parts.append(f"candidate_{candidate_id}")
    filename = "_".join(filename_parts) + ".md"

    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# LLM Interaction Log\n\n")
        f.write(f"- **Generation**: {generation}\n")
        f.write(f"- **Phase**: {phase}\n")
        if candidate_id is not None:
            f.write(f"- **Candidate Info**: {candidate_id}\n")
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"## Prompt (Sent to LLM)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"```text\n{prompt}\n```\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"## Response (From LLM)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"```text\n{response}\n```\n")
```

### 5.2 埋点位置

| 埋点编号 | 位置 | 阶段 | 候选标识规则 |
|----------|------|------|--------------|
| 埋点1 | `generate_candidates()` → CoT分析后 | `Phase1_CoT_Analysis` | 无候选ID |
| 埋点2 | `generate_candidates()` → 初始生成后 | `Phase2_Initial_Generation` | `candidate_{idx}`（并发数组索引） |
| 埋点3 | `_call_evoleap()` 闭包内部 | `Phase3_EvoLeap` | `worker{idx}_{op_type}`（线程索引+算子类型） |
| 埋点4 | `analyze_results()` → 反思生成成功后 | `Phase4_Reflection` | `best_code_diagnosis` |

### 5.3 阶段三的特殊处理

阶段三（EvoLeap变异）由于涉及并发多线程执行，对 `_call_evoleap` 函数进行了签名修改，增加了 `worker_idx` 参数以区分不同的并发线程：

```python
# 修改前
def _call_evoleap(op_type: str) -> str:
    ...

# 修改后
def _call_evoleap(op_type: str, worker_idx: int) -> str:
    # 在返回前保存交互记录
    self._save_llm_interaction(
        generation, "Phase3_EvoLeap", prompt,
        response_text, candidate_id=f"worker{worker_idx}_{op_type}"
    )
    return response_text

# 调用时传入线程索引
future_to_op = {
    executor.submit(_call_evoleap, op, idx): op
    for idx, op in enumerate(assigned_operators)
}
```

## 六、测试验证

测试文件位于 `tests/agent/test_llm_interaction_logging.py`，包含以下测试用例：

| 测试用例 | 说明 |
|----------|------|
| `test_save_llm_interaction_basic` | 验证基础保存功能 |
| `test_save_llm_interaction_multiple_generations` | 验证多代保存功能 |
| `test_save_llm_interaction_phases` | 验证不同阶段保存 |
| `test_save_llm_interaction_without_candidate_id` | 验证不提供候选ID的情况 |
| `test_save_llm_interaction_special_characters` | 验证特殊字符处理 |
| `test_directory_structure` | 验证目录结构正确性 |

运行测试：

```bash
python tests/agent/test_llm_interaction_logging.py
```

预期输出：

```
================================================================================
LLM交互记录保存功能测试
================================================================================

[TEST] test_save_llm_interaction_basic: 测试基础保存功能
  [OK] 文件已生成: 20260408_163910_Phase1_CoT_Analysis_candidate_test_001.md
  [OK] 文件路径: ...\llm_interactions\generation_0\...
[PASS] test_save_llm_interaction_basic

...（其他测试省略）...

================================================================================
测试完成: 6 通过, 0 失败
================================================================================
```

## 七、配置说明

功能使用现有的日志配置，无需额外配置项：

```yaml
logging:
  save_dir: 'experiments'  # 交互记录将保存在 experiments/llm_interactions/ 下
```

## 八、使用场景

1. **学术复现**: 记录每次LLM决策的完整上下文，便于复现实验结果
2. **调试分析**: 当某个候选代码性能异常时，可追溯LLM生成该代码时的完整提示词和上下文
3. **知识挖掘**: 分析LLM在不同阶段的响应模式，提取有效的奖励函数设计策略
4. **异常诊断**: 当反思生成失败时，可查看历史交互记录定位问题根因
5. **版本对比**: 对比不同代数、不同阶段LLM响应的变化趋势

## 九、注意事项

1. **存储空间**: 每次LLM调用都会生成一个文件，长期运行可能产生大量文件，建议定期归档
2. **敏感信息**: 文件中包含完整的提示词和响应内容，注意保护API密钥和敏感业务信息
3. **并发安全**: 阶段三使用线程池并发执行，`_save_llm_interaction` 方法本身是文件I/O操作，不涉及共享状态，无需额外加锁
4. **文件编码**: 统一使用UTF-8编码，确保中文字符正确保存
