"""
LLM交互记录保存功能测试
测试 _save_llm_interaction 方法的正确性

Author: LEMS Project
Date: 2026-04-08
"""

import sys
import os
import shutil
import tempfile
import yaml
import io

# 设置标准输出编码为UTF-8（解决Windows下的中文显示问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_reward_agent.agent.reward_design_agent import RewardDesignAgent


def create_test_config(save_dir):
    """创建测试用配置"""
    config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': 'test-key',
            'timeout': 10,
            'max_retries': 1
        },
        'generation': {
            'num_candidates': 4,
            'temperature': 0.8,
            'max_tokens': 2000
        },
        'reflection': {
            'temperature': 0.3,
            'max_tokens': 1500
        },
        'training': {
            'episode_num': 10,
            'parallel_workers': 2,
            'timeout': 300,
            'use_gpu': False
        },
        'logging': {
            'save_dir': save_dir
        },
        'fitness': {
            'weights': {
                'success_rate': 1.0,
                'capture_time': -0.001,
                'formation_quality': 0.3,
                'collision_penalty': -0.5
            }
        }
    }
    return config


def test_save_llm_interaction_basic():
    """测试基础保存功能"""
    print("[TEST] test_save_llm_interaction_basic: 测试基础保存功能")
    
    test_dir = tempfile.mkdtemp(prefix="llm_interaction_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 测试数据
        generation = 0
        phase = "Phase1_CoT_Analysis"
        prompt = "这是一个测试提示词\n包含多行内容\n用于测试保存功能"
        response = "这是一个测试响应\n同样包含多行内容"
        candidate_id = "test_001"
        
        # 调用保存方法
        agent._save_llm_interaction(generation, phase, prompt, response, candidate_id)
        
        # 验证文件是否生成
        interaction_dir = os.path.join(
            test_dir, 'llm_interactions', f'generation_{generation}'
        )
        assert os.path.exists(interaction_dir), f"交互记录目录未创建: {interaction_dir}"
        
        # 验证目录中是否有文件
        files = os.listdir(interaction_dir)
        assert len(files) > 0, "交互记录目录为空"
        
        # 验证文件名包含相关信息
        md_files = [f for f in files if f.endswith('.md')]
        assert len(md_files) > 0, "没有生成Markdown文件"
        
        # 读取并验证文件内容
        file_path = os.path.join(interaction_dir, md_files[0])
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证关键内容
        assert "LLM Interaction Log" in content, "文件头部不正确"
        assert f"**Generation**: {generation}" in content, "代数信息缺失"
        assert f"**Phase**: {phase}" in content, "阶段信息缺失"
        assert f"**Candidate Info**: {candidate_id}" in content, "候选ID信息缺失"
        assert prompt in content, "提示词内容缺失"
        assert response in content, "响应内容缺失"
        
        print(f"  [OK] 文件已生成: {md_files[0]}")
        print(f"  [OK] 文件路径: {file_path}")
        print("[PASS] test_save_llm_interaction_basic")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        # 清理测试目录
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_save_llm_interaction_multiple_generations():
    """测试多代保存功能"""
    print("[TEST] test_save_llm_interaction_multiple_generations: 测试多代保存功能")
    
    test_dir = tempfile.mkdtemp(prefix="llm_multi_gen_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 测试不同代数
        generations = [0, 1, 2, 3]
        
        for gen in generations:
            phase = f"Phase2_Initial_Generation"
            prompt = f"第{gen}代测试提示词"
            response = f"第{gen}代测试响应"
            candidate_id = str(gen)
            
            agent._save_llm_interaction(gen, phase, prompt, response, candidate_id)
        
        # 验证每个代数都有对应的目录
        for gen in generations:
            interaction_dir = os.path.join(
                test_dir, 'llm_interactions', f'generation_{gen}'
            )
            assert os.path.exists(interaction_dir), f"第{gen}代目录未创建"
            
            files = os.listdir(interaction_dir)
            assert len(files) > 0, f"第{gen}代目录为空"
        
        print(f"  [OK] 已创建 {len(generations)} 个代的目录")
        print("[PASS] test_save_llm_interaction_multiple_generations")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_save_llm_interaction_phases():
    """测试不同阶段保存"""
    print("[TEST] test_save_llm_interaction_phases: 测试不同阶段保存")
    
    test_dir = tempfile.mkdtemp(prefix="llm_phases_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 测试各个阶段
        phases = [
            ("Phase1_CoT_Analysis", None),
            ("Phase2_Initial_Generation", "0"),
            ("Phase2_Initial_Generation", "1"),
            ("Phase3_EvoLeap", "worker0_F1"),
            ("Phase3_EvoLeap", "worker1_F2"),
            ("Phase4_Reflection", "best_code_diagnosis"),
        ]
        
        generation = 0
        
        for phase, candidate_id in phases:
            prompt = f"{phase} 测试提示词"
            response = f"{phase} 测试响应"
            
            agent._save_llm_interaction(
                generation, phase, prompt, response, candidate_id
            )
        
        # 验证目录和文件
        interaction_dir = os.path.join(
            test_dir, 'llm_interactions', f'generation_{generation}'
        )
        files = os.listdir(interaction_dir)
        
        assert len(files) == len(phases), \
            f"预期 {len(phases)} 个文件，实际 {len(files)} 个"
        
        # 验证每个阶段的文件都能找到
        for phase, candidate_id in phases:
            found = any(phase in f for f in files)
            assert found, f"未找到阶段 {phase} 的文件"
        
        print(f"  [OK] 已保存 {len(phases)} 个阶段的交互记录")
        print("[PASS] test_save_llm_interaction_phases")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_save_llm_interaction_without_candidate_id():
    """测试不提供候选ID的情况"""
    print("[TEST] test_save_llm_interaction_without_candidate_id: 测试不提供候选ID")
    
    test_dir = tempfile.mkdtemp(prefix="llm_no_candidate_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 不提供candidate_id
        generation = 0
        phase = "Phase1_CoT_Analysis"
        prompt = "测试提示词"
        response = "测试响应"
        
        agent._save_llm_interaction(generation, phase, prompt, response)
        
        # 验证文件生成
        interaction_dir = os.path.join(
            test_dir, 'llm_interactions', f'generation_{generation}'
        )
        files = os.listdir(interaction_dir)
        
        assert len(files) > 0, "文件未生成"
        
        # 验证文件内容不包含候选ID
        file_path = os.path.join(interaction_dir, files[0])
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "**Candidate Info**" not in content, \
            "不应该包含候选ID信息"
        
        print("[PASS] test_save_llm_interaction_without_candidate_id")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_save_llm_interaction_special_characters():
    """测试特殊字符处理"""
    print("[TEST] test_save_llm_interaction_special_characters: 测试特殊字符处理")
    
    test_dir = tempfile.mkdtemp(prefix="llm_special_char_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 包含特殊字符的内容
        generation = 0
        phase = "Phase1_CoT_Analysis"
        prompt = "测试提示词\n包含特殊字符: {}[]#$\n代码块 ```python\nprint('hello')\n```"
        response = "测试响应\n包含中文测试\n特殊符号: @#$%^&*()"
        candidate_id = "special_001"
        
        agent._save_llm_interaction(generation, phase, prompt, response, candidate_id)
        
        # 验证文件生成和读取
        interaction_dir = os.path.join(
            test_dir, 'llm_interactions', f'generation_{generation}'
        )
        files = os.listdir(interaction_dir)
        
        assert len(files) > 0, "文件未生成"
        
        # 读取并验证
        file_path = os.path.join(interaction_dir, files[0])
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证prompt和response被正确包装在```text```中
        assert "```text" in content, "代码块格式不正确"
        
        print("[PASS] test_save_llm_interaction_special_characters")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_directory_structure():
    """测试目录结构是否正确"""
    print("[TEST] test_directory_structure: 测试目录结构")
    
    test_dir = tempfile.mkdtemp(prefix="llm_dir_structure_test_")
    config = create_test_config(test_dir)
    config_path = os.path.join(test_dir, 'test_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        agent = RewardDesignAgent(config_path=config_path)
        
        # 保存一个交互记录
        agent._save_llm_interaction(
            generation=5,
            phase="Phase1_CoT_Analysis",
            prompt="测试",
            response="测试",
            candidate_id="test"
        )
        
        # 验证预期的目录结构
        expected_dir = os.path.join(
            test_dir,  # save_dir
            'llm_interactions',  # 固定子目录
            'generation_5'  # 代数目录
        )
        
        assert os.path.exists(expected_dir), \
            f"目录结构不正确: 预期 {expected_dir}"
        
        # 验证 llm_interactions 目录在正确的位置
        llm_interactions_dir = os.path.join(test_dir, 'llm_interactions')
        assert os.path.exists(llm_interactions_dir), \
            "llm_interactions 目录未在正确位置创建"
        
        print(f"  [OK] 目录结构: test_dir/llm_interactions/generation_X/")
        print(f"  [OK] 实际路径: {expected_dir}")
        print("[PASS] test_directory_structure")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("LLM交互记录保存功能测试")
    print("=" * 80)
    print()
    
    tests = [
        test_save_llm_interaction_basic,
        test_save_llm_interaction_multiple_generations,
        test_save_llm_interaction_phases,
        test_save_llm_interaction_without_candidate_id,
        test_save_llm_interaction_special_characters,
        test_directory_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
