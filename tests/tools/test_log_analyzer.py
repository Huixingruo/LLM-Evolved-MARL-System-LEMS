"""
log_analyzer.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_reward_agent.tools.log_analyzer import LogAnalyzer


def test_log_analyzer_initialization():
    """测试 LogAnalyzer 初始化"""
    analyzer = LogAnalyzer()
    assert analyzer is not None
    assert 'weights' in analyzer.fitness_config
    print("[OK] test_log_analyzer_initialization")


def test_log_analyzer_with_custom_config():
    """测试自定义配置"""
    custom_config = {
        'weights': {
            'success_rate': 2.0,
            'capture_time': -0.002,
            'formation_quality': 0.5
        },
        'normalize': {
            'max_capture_time': 100,
            'min_success_rate': 0.0,
            'max_success_rate': 1.0
        }
    }

    analyzer = LogAnalyzer(fitness_config=custom_config)
    assert analyzer.fitness_config['weights']['success_rate'] == 2.0
    print("[OK] test_log_analyzer_with_custom_config")


def test_calculate_fitness():
    """测试 Fitness 计算"""
    analyzer = LogAnalyzer()

    metrics = {
        'success_rate': 0.75,
        'avg_capture_time': 50.0,
        'reward_components': {},
        'collaboration_metrics': {}
    }

    fitness = analyzer.calculate_fitness(metrics)
    assert isinstance(fitness, float)
    # 默认权重: success_rate * 1.0 + capture_time * (-0.001) = 0.75 - 0.05 = 0.70
    print(f"[OK] test_calculate_fitness: fitness = {fitness:.4f}")


def test_calculate_fitness_with_components():
    """测试带奖励分量的 Fitness 计算"""
    analyzer = LogAnalyzer()

    metrics = {
        'success_rate': 0.80,
        'avg_capture_time': 40.0,
        'reward_components': {
            'collision_penalty': {'mean': -0.5, 'std': 0.2}
        },
        'collaboration_metrics': {
            'formation_quality': {'mean': 0.6}
        }
    }

    fitness = analyzer.calculate_fitness(metrics)
    assert isinstance(fitness, float)
    print(f"[OK] test_calculate_fitness_with_components: fitness = {fitness:.4f}")


def test_generate_analysis_report():
    """测试分析报告生成"""
    analyzer = LogAnalyzer()

    metrics = {
        'fitness': 0.75,
        'success_rate': 0.80,
        'avg_capture_time': 45.0,
        'reward_components': {
            'distance_reward': {'mean': -1.2, 'std': 0.4},
            'collision_penalty': {'mean': -0.3, 'std': 0.1}
        },
        'collaboration_metrics': {
            'formation_quality': {'mean': 0.6}
        }
    }

    report = analyzer.generate_analysis_report(metrics)
    assert '综合评分' in report
    assert '任务性能' in report
    assert '奖励分量统计' in report
    print("[OK] test_generate_analysis_report")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 LogAnalyzer 模块")
    print("=" * 80)
    print()

    tests = [
        test_log_analyzer_initialization,
        test_log_analyzer_with_custom_config,
        test_calculate_fitness,
        test_calculate_fitness_with_components,
        test_generate_analysis_report,
    ]

    passed = 0
    failed = 0

    for test in tests:
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
