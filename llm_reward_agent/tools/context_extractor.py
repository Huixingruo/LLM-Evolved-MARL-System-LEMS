"""
环境上下文提取器 - 智能环境上下文提取器
特点：
1. 运行时反射：真实运行环境以获取准确的 space 维度
2. AST 代码清洗：自动剔除渲染代码、非核心 import，保留物理逻辑

Author: LEMS Project
Date: 2026-02-11
Version: 2.0 (Based on Professor's Insights)
"""

import ast
import inspect
import sys
import os
import importlib.util
import numpy as np
from typing import Dict, List, Any, Optional


class LLMFriendlyContextExtractor:
    """智能环境上下文提取器"""

    def __init__(self, env_file_path: str):
        """
        初始化提取器

        Args:
            env_file_path: 环境文件路径
        """
        self.file_path = env_file_path
        self.module_name = os.path.basename(env_file_path).replace('.py', '')

        # 需要保留的类和方法
        self.keep_classes = ['Custom_raw_env', 'Scenario']
        self.keep_methods = [
            'make_world', 'reset_world', 'reward', 'observation',
            '_execute_world_step', 'check_capture_condition', '_build_global_state'
        ]

        # 需要移除的渲染相关函数
        self.render_blacklist = [
            'render', 'draw', 'enable_render', 'render_matplotlib',
            'draw_grid_and_axes', '_draw'
        ]

        # 需要移除的导入
        self.import_blacklist = ['pygame', 'matplotlib']

    def get_runtime_info(self) -> Dict[str, Any]:
        """
        通过动态加载模块并实例化环境，获取真实的维度信息

        Returns:
            dict: 包含观测空间、动作空间等真实维度信息
        """
        try:
            # 读取源代码
            with open(self.file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # 替换相对导入为绝对导入
            code = code.replace('from .custom_agents_dynamics import CustomWorld',
                              'from MADDPG.envs.custom_agents_dynamics import CustomWorld')
            code = code.replace('from . import reward_function',
                              'from MADDPG.envs import reward_function')

            # 获取envs目录
            envs_dir = os.path.dirname(os.path.abspath(self.file_path))
            project_root = os.path.dirname(os.path.dirname(envs_dir))

            # 添加到sys.path
            if envs_dir not in sys.path:
                sys.path.insert(0, envs_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # 创建模块命名空间
            namespace = {
                '__name__': self.module_name,
                '__file__': self.file_path,
                '__builtins__': __builtins__,
            }

            # 编译并执行
            compiled_code = compile(code, self.file_path, 'exec')
            exec(compiled_code, namespace)

            # 获取环境类
            env_class = namespace.get('Custom_raw_env')

            if env_class is None:
                return {"error": "未找到 Custom_raw_env 类"}

            # 实例化环境
            try:
                env = env_class(
                    num_good=1,
                    num_adversaries=3,
                    num_obstacles=0,
                    continuous_actions=True,
                    render_mode=None
                )
                env.reset()

                info = {
                    "env_class_name": env_class.__name__,
                    "agents": [],
                    "observation_structure": {},
                    "action_structure": {},
                    "physical_constants": {
                        "world_size": getattr(env, 'world_size', 'N/A'),
                        "max_force": getattr(env, 'max_force', 'N/A'),
                        "capture_threshold": getattr(env, 'capture_threshold', 'N/A'),
                        "max_cycles": getattr(env, 'max_cycles', 'N/A'),
                    }
                }

                # 提取每个智能体的空间信息
                for agent_name in env.agents:
                    obs_space = env.observation_space(agent_name)
                    act_space = env.action_space(agent_name)

                    info["agents"].append(agent_name)
                    info["observation_structure"][agent_name] = {
                        "shape": str(obs_space.shape),
                        "dtype": str(obs_space.dtype),
                        "low": float(np.min(obs_space.low)) if hasattr(obs_space, 'low') else -np.inf,
                        "high": float(np.max(obs_space.high)) if hasattr(obs_space, 'high') else np.inf,
                    }
                    info["action_structure"][agent_name] = {
                        "type": type(act_space).__name__,
                        "shape": str(act_space.shape) if hasattr(act_space, 'shape') else "N/A",
                        "dtype": str(act_space.dtype) if hasattr(act_space, 'dtype') else "N/A",
                    }

                # 清理环境
                if hasattr(env, 'close'):
                    try:
                        env.close()
                    except:
                        pass

                return info

            except Exception as e:
                return {"error": f"环境实例化失败: {str(e)}"}

        except Exception as e:
            return {"error": f"模块加载失败: {str(e)}"}

    def clean_code_with_ast(self) -> str:
        """
        使用AST解析代码，移除渲染函数，保留物理逻辑

        Returns:
            str: 清洗后的代码
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # 获取外部类的属性
            keep_classes = self.keep_classes
            render_blacklist = self.render_blacklist
            import_blacklist = self.import_blacklist

            class CodeCleaner(ast.NodeTransformer):
                """代码清洗器"""

                def __init__(self, keep_classes, render_blacklist, import_blacklist):
                    super().__init__()
                    self.keep_classes = keep_classes
                    self.render_blacklist = render_blacklist
                    self.import_blacklist = import_blacklist

                def visit_Import(self, node):
                    """移除渲染相关的import"""
                    names = [n for n in node.names
                            if not any(black in n.name.lower() or n.name.lower() in self.import_blacklist
                                      for black in self.import_blacklist)]
                    if names:
                        node.names = [n for n in node.names
                                     if not any(black in n.name.lower() for black in self.import_blacklist)]
                        return node if node.names else None
                    return None

                def visit_ImportFrom(self, node):
                    """移除渲染相关的from导入"""
                    if node.module and any(black in node.module.lower() for black in self.import_blacklist):
                        return None
                    return node

                def visit_ClassDef(self, node):
                    """只保留核心类"""
                    if node.name in self.keep_classes:
                        self.generic_visit(node)
                        return node
                    return None

                def visit_FunctionDef(self, node):
                    """移除渲染相关函数"""
                    if any(black in node.name for black in self.render_blacklist):
                        return None
                    return node

                def visit_Assign(self, node):
                    """处理赋值语句"""
                    # 移除渲染相关的全局变量赋值
                    if isinstance(node.targets[0], ast.Name):
                        if any(black in node.targets[0].id for black in self.render_blacklist):
                            return None
                    return node

                def visit_If(self, node):
                    """移除 if __name__ == '__main__': 代码块"""
                    # 检测 if __name__ == '__main__': 模式
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__' and
                        len(node.test.comparators) == 1 and
                        isinstance(node.test.comparators[0], ast.Constant) and
                        node.test.comparators[0].value == '__main__'):
                        return None  # 移除整个 if __name__ == '__main__' 块
                    self.generic_visit(node)
                    return node

                def visit_Module(self, node):
                    """移除不需要的导入和方法"""
                    new_body = []
                    for item in node.body:
                        # 1. 移除所有导入语句（56-64行）
                        if isinstance(item, (ast.Import, ast.ImportFrom)):
                            continue
                        # 2. 移除 Custom_raw_env 类中的以下方法（115-252行）：
                        #    reset, reset_world, _execute_world_step, step, check_capture_condition
                        if isinstance(item, ast.ClassDef) and item.name == 'Custom_raw_env':
                            new_methods = []
                            methods_to_remove = ['reset', 'reset_world', '_execute_world_step', 'step', 'check_capture_condition']
                            for child in ast.iter_child_nodes(item):
                                if isinstance(child, ast.FunctionDef):
                                    # 保留核心方法：__init__, _init_spaces, _set_action
                                    if child.name in methods_to_remove:
                                        continue
                                new_methods.append(child)
                            # 重建类节点
                            new_item = ast.ClassDef(
                                name=item.name,
                                bases=item.bases,
                                keywords=item.keywords,
                                body=new_methods,
                                decorator_list=item.decorator_list
                            )
                            ast.copy_location(new_item, item)
                            new_body.append(new_item)
                        # 3. 移除 Scenario 类中的 reset_world 和 benchmark_data 方法（311-334行）
                        elif isinstance(item, ast.ClassDef) and item.name == 'Scenario':
                            new_methods = []
                            for child in ast.iter_child_nodes(item):
                                if isinstance(child, ast.FunctionDef):
                                    if child.name in ['reset_world', 'benchmark_data']:
                                        continue
                                new_methods.append(child)
                            # 重建类节点
                            new_item = ast.ClassDef(
                                name=item.name,
                                bases=item.bases,
                                keywords=item.keywords,
                                body=new_methods,
                                decorator_list=item.decorator_list
                            )
                            ast.copy_location(new_item, item)
                            new_body.append(new_item)
                        else:
                            new_body.append(item)
                    node.body = new_body
                    self.generic_visit(node)
                    return node

            cleaner = CodeCleaner(keep_classes, render_blacklist, import_blacklist)
            cleaned_tree = cleaner.visit(tree)
            ast.fix_missing_locations(cleaned_tree)

            return ast.unparse(cleaned_tree)

        except Exception as e:
            return f"代码清洗失败: {str(e)}"

    def generate_prompt_context(self) -> str:
        """
        生成LLM友好的提示上下文

        Returns:
            str: 格式化的提示上下文
        """
        runtime_info = self.get_runtime_info()
        cleaned_code = self.clean_code_with_ast()

        # 构建物理常量说明
        physical_constants = runtime_info.get('physical_constants', {})

        context = f"""
|# Environment Context for Reward Design

## 1. Runtime Spaces Analysis (Ground Truth)

```
{self._format_runtime_info(runtime_info)}
```

### 观测向量切片语义 (CRITICAL for Reward Design)
---
本环境使用 Simple Tag 场景，观测向量按以下顺序拼接：
```python
return np.concatenate([norm_self_vel] + [norm_self_pos] + entity_pos + other_pos + other_vel])
```

**追捕者 (adversary) 的观测向量索引：**
| 索引范围 | 语义 | 物理含义 |
|---------|------|---------|
| [0:2] | Self Velocity | 自身速度 (vx, vy)，归一化到 [-1, 1] |
| [2:4] | Self Position | 自身绝对位置 (x, y)，归一化到 [-1, 1] |
| [4:6] | Rel Pos (Adversary 1) | 队友1相对于自己的位置 |
| [6:8] | Rel Pos (Adversary 2) | 队友2相对于自己的位置 |
| [8:10] | **Rel Pos (Prey)** | **猎物相对于自己的位置 ← 核心！** |
| [10:12] | Prey Velocity | 猎物的速度，归一化到 [-1, 1] |

**逃跑者 (agent) 的观测向量索引：**
| 索引范围 | 语义 | 物理含义 |
|---------|------|---------|
| [0:2] | Self Velocity | 自身速度 |
| [2:4] | Self Position | 自身位置 |
| [4:6] | Rel Pos (Adv 1) | 追捕者1相对于自己的位置 |
| [6:8] | Rel Pos (Adv 2) | 追捕者2相对于自己的位置 |
| [8:10] | Rel Pos (Adv 3) | 追捕者3相对于自己的位置 |
---

## 2. Core Environment Logic (Rendering Stripped)

```python
{cleaned_code}
```

## 3. Key Physical Constants (Ground Truth)

|| 常量名 | 值 | 物理含义 |
||-------|-----|---------|
|| world_size | {physical_constants.get('world_size', 'N/A')} | 世界边界 (+/-world_size) |
|| max_force | {physical_constants.get('max_force', 'N/A')} | 最大作用力 |
|| capture_threshold | {physical_constants.get('capture_threshold', 'N/A')} | 围捕判定距离 |
|| dim_p | 2 | 位置空间维度 (x, y) |
|| dim_c | 0 | 通信通道维度 |

## 4. Task Description

设计追捕者(adversary)的奖励函数：
- **目标**: 3个追捕者协同围捕1个逃跑者
- **成功条件**: 所有追捕者进入逃跑者的capture_threshold范围内
- **关键**: 设计协作策略，使追捕者形成包围圈

逃跑者的奖励函数已固定，不需要重新设计。
"""

        return context

    def _format_runtime_info(self, info: Dict) -> str:
        """格式化运行时信息"""
        if "error" in info:
            return f"Error: {info['error']}"

        lines = []
        lines.append(f"环境类: {info.get('env_class_name', 'N/A')}")
        lines.append(f"智能体数量: {len(info.get('agents', []))}")
        lines.append("")

        lines.append("动作空间 (Ground Truth):")
        for agent_name in info.get('agents', []):
            act_info = info.get('action_structure', {}).get(agent_name, {})
            lines.append(f"  {agent_name}: {act_info.get('type', 'N/A')} - shape={act_info.get('shape', 'N/A')}")

        lines.append("")
        lines.append("观测空间 (Ground Truth):")
        for agent_name in info.get('agents', []):
            obs_info = info.get('observation_structure', {}).get(agent_name, {})
            lines.append(f"  {agent_name}: shape={obs_info.get('shape', 'N/A')}, dtype={obs_info.get('dtype', 'N/A')}")

        return '\n'.join(lines)

    def extract_key_methods(self) -> Dict[str, str]:
        """
        提取关键方法的代码

        Returns:
            dict: 方法名 -> 方法代码
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)
            key_methods = {}

            class MethodExtractor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None
                    self.methods = {}

                def visit_ClassDef(self, node):
                    if node.name in self.keep_classes:
                        self.current_class = node.name
                        self.generic_visit(node)
                        self.current_class = None

                def visit_FunctionDef(self, node):
                    if (self.current_class and
                        node.name in self.keep_methods and
                        self.current_class == 'Scenario'):
                        # 获取方法源代码
                        lines = source.split('\n')
                        start = node.lineno - 1
                        # 找到方法结束位置
                        end = start
                        for i in range(start + 1, len(lines)):
                            if lines[i].startswith(' ' * (len(lines[start]) - len(lines[start].lstrip()) + 4)) and 'def ' in lines[i]:
                                break
                            if lines[i].strip() and not lines[i].startswith(' ' * (len(lines[start]) - len(lines[start].lstrip()) + 4)):
                                break
                            end = i + 1

                        method_code = '\n'.join(lines[start:end])
                        key_methods[node.name] = method_code

            extractor = MethodExtractor()
            extractor.visit(tree)

            return key_methods

        except Exception as e:
            return {"error": str(e)}


# ========================================
# 兼容性包装类
# ========================================

class EnvironmentContextExtractor(LLMFriendlyContextExtractor):
    """
    兼容性包装类 - 保持原有接口

    向后兼容原有代码，同时提供新功能
    """

    def __init__(self):
        # 查找默认环境文件
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'envs', 'simple_tag_env.py'
        )
        super().__init__(default_path)

    def extract_skeleton(self, env_file_path: str) -> Dict[str, Any]:
        """
        提取环境代码骨架（原有接口）

        Args:
            env_file_path: 环境文件路径

        Returns:
            dict: 环境上下文信息
        """
        # 使用运行时信息
        runtime_info = self.get_runtime_info()

        return {
            "env_name": os.path.basename(env_file_path).replace('.py', ''),
            "file_path": env_file_path,
            "runtime_info": runtime_info,
            "cleaned_code": self.clean_code_with_ast()
        }

    def format_for_llm(self, context: Dict) -> str:
        """
        格式化上下文信息为LLM友好的文本（原有接口）

        Args:
            context: 环境上下文字典

        Returns:
            str: 格式化后的文本
        """
        return self.generate_prompt_context()

    def estimate_token_count(self, text: str) -> int:
        """
        估算文本的Token数量

        Args:
            text: 输入文本

        Returns:
            int: 估计的Token数量
        """
        return len(text) // 4


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing LLMFriendlyContextExtractor")
    print("=" * 80)

    # 查找测试文件
    search_paths = [
        "MADDPG/envs/simple_tag_env.py",
        "../MADDPG/envs/simple_tag_env.py",
        "../../MADDPG/envs/simple_tag_env.py",
        os.path.join(os.path.dirname(__file__), "../../MADDPG/envs/simple_tag_env.py"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "MADDPG/envs/simple_tag_env.py"),
    ]

    test_file = None
    for path in search_paths:
        if os.path.exists(path):
            test_file = path
            break

    if test_file:
        print(f"\nUsing test file: {test_file}")

        # 创建提取器
        extractor = LLMFriendlyContextExtractor(test_file)

        print(f"\n[1/3] Testing runtime info extraction...")
        runtime_info = extractor.get_runtime_info()
        if "error" not in runtime_info:
            print(f"  [OK] Runtime info extracted successfully")
            print(f"  Environment class: {runtime_info.get('env_class_name')}")
            print(f"  Agents: {runtime_info.get('agents')}")
        else:
            print(f"  [FAIL] {runtime_info['error']}")

        print(f"\n[2/3] Testing code cleaning...")
        cleaned = extractor.clean_code_with_ast()
        if "error" not in cleaned:
            print(f"  [OK] Code cleaning successful (lines: {cleaned.count(chr(10))+1})")
        else:
            print(f"  [FAIL] {cleaned}")

        print(f"\n[3/3] Testing full context generation...")
        full_context = extractor.generate_prompt_context()
        if "error" not in full_context:
            print(f"  [OK] Context generated (characters: {len(full_context)})")
            # 保存到文件
            output_file = "env_context_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_context)
            print(f"  Saved to: {output_file}")
        else:
            print(f"  [FAIL] {full_context}")

        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)
    else:
        print("Test file not found: simple_tag_env.py")
        print("\nSearch paths:")
        for path in search_paths:
            print(f"  - {path} (exists: {os.path.exists(path)})")
