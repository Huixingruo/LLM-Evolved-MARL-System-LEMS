"""
LLM接口封装
统一的LLM调用接口，支持多种模型

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import time
import concurrent.futures
from typing import List, Dict, Optional
from openai import OpenAI, APIError, APITimeoutError


class LLMInterface:
    """统一的LLM调用接口，支持多种模型"""
    
    def __init__(self,
                 provider: str = "openai",
                 model_name: str = "gpt-5.1",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = "https://api.vectorengine.ai/v1",
                 timeout: int = 120,
                 max_retries: int = 3):
        """
        初始化LLM接口
        
        Args:
            provider: LLM提供商 (openai, anthropic, zhipu等)
            model_name: 模型名称
            api_key: API密钥（如果为None则从环境变量读取）
            base_url: 自定义API端点
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.provider = provider
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 处理API密钥
        if api_key is None:
            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "zhipu":
                api_key = os.getenv("ZHIPU_API_KEY")
            elif provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            raise ValueError(f"API密钥未设置。请设置环境变量或传入api_key参数。")
        
        # 初始化客户端
        if provider == "openai":
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
        elif provider == "deepseek":
            # DeepSeek使用OpenAI兼容接口
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.deepseek.com/v1",
                timeout=timeout
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
        
        print(f"✅ LLM接口初始化完成: {provider}/{model_name}")
    
    def generate(self, 
             prompt: str, 
             n: int = 1, 
             temperature: float = 0.7,
             max_tokens: int = 2000,
             system_message: Optional[str] = None) -> List[str]:
        """
        生成N个不同的回复（支持自动多线程并发，兼容不支持 n 参数的模型）
        """
        # =========================================================
        # 内部函数：定义单次调用的逻辑
        # =========================================================
        def _single_call():
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=1,  # 强制单次只请求1个
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )
                # # 兼容性处理：如果返回的直接是字符串，说明API返回了纯文本
                # if isinstance(response, str):
                #     return response
                # # ===================
                return response.choices[0].message.content
            except Exception as err:
                print(f"⚠️ LLM API调用失败: {err}")
                import traceback
                traceback.print_exc()
                return None

        # =========================================================
        # 策略 A: 如果 n=1，直接调用 (最快)
        # =========================================================
        if n == 1:
            res = _single_call()
            return [res] if res else []

        # =========================================================
        # 策略 B: 如果 n > 1，使用多线程并发 (兼容性最强)
        # =========================================================
        print(f"🚀 正在并发请求 {n} 个回复 (多线程模式)...")
        results = []
        
        # 使用线程池并发发起 N 个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            # 提交任务
            futures = [executor.submit(_single_call) for _ in range(n)]
            
            # 获取结果
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
        
        print(f"✅ LLM生成成功: 获取到 {len(results)}/{n} 个回复")
        return results
    
    def analyze(self, 
                prompt: str, 
                temperature: float = 0.3,
                max_tokens: int = 1500,
                system_message: Optional[str] = None) -> str:
        """
        单次分析调用（用于Reflection）
        
        Args:
            prompt: 分析提示词
            temperature: 温度参数（较低保证准确性）
            max_tokens: 最大Token数
            system_message: 系统提示词
        
        Returns:
            str: 分析结果
        """
        results = self.generate(
            prompt=prompt,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message
        )

        # 如果返回空列表，抛出异常让上层处理
        if not results:
            raise RuntimeError(
                f"LLM analyze调用失败: generate返回空列表\n"
                f"Prompt长度: {len(prompt)} chars"
            )

        return results[0]
    
    def estimate_cost(self, 
                      input_tokens: int, 
                      output_tokens: int) -> float:
        """
        估算API调用成本（美元）
        
        Args:
            input_tokens: 输入Token数
            output_tokens: 输出Token数
        
        Returns:
            float: 估算成本（USD）
        """
        # 价格表（2026年估算，实际请查询最新价格）
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
            "deepseek-chat": {"input": 0.001 / 1000, "output": 0.002 / 1000},
        }
        
        model_price = pricing.get(self.model_name, {"input": 0.01/1000, "output": 0.03/1000})
        
        cost = (input_tokens * model_price["input"] + 
                output_tokens * model_price["output"])
        
        return cost


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试LLM接口...")
    
    # 测试前请设置环境变量: export OPENAI_API_KEY=your_key
    # 或者直接传入api_key参数
    
    try:
        # 初始化接口
        llm = LLMInterface(
            provider="openai",
            model_name="gpt-5.1", 
            # api_key="your_key_here"  # 或者从环境变量读取
        )
        
        # 测试生成
        print("\n=== 测试代码生成 ===")
        prompt = """请编写一个简单的Python函数，计算两个数的和。
只输出代码，不要有任何解释。

```python
"""
        
        results = llm.generate(
            prompt=prompt,
            n=1,
            temperature=0.7,
            max_tokens=200
        )
        
        for i, result in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(result[:1000])  # 只显示前200字符
        
        # # 测试分析
        # print("\n=== 测试分析功能 ===")
        # analysis_prompt = "分析以下代码的时间复杂度: def foo(n): return sum(range(n))"
        
        # analysis = llm.analyze(
        #     prompt=analysis_prompt,
        #     temperature=0.3
        # )
        
        # print(f"\n分析结果:\n{analysis[:1000]}")
        
        # # 估算成本
        # print("\n=== 成本估算 ===")
        # cost = llm.estimate_cost(input_tokens=500, output_tokens=1000)
        # print(f"估算成本: ${cost:.6f}")
        
        # print("\n✅ LLM接口测试完成！")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示：请确保已设置API密钥环境变量")
