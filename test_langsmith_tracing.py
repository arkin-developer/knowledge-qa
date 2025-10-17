#!/usr/bin/env python3
"""
LangSmith 可观测性测试脚本

运行此脚本将演示完整的调用链追踪功能，包括：
- Agent 节点执行追踪
- LLM 调用追踪  
- 向量搜索追踪
- 响应时间和性能指标

使用方法：
uv run python test_langsmith_tracing.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.knowledge_qa.agent import KnowledgeQAAgent
from src.knowledge_qa.log_manager import log


def test_langsmith_tracing():
    """测试 LangSmith 可观测性功能"""
    print("=" * 80)
    print("LangSmith 可观测性功能测试")
    print("=" * 80)
    
    # 初始化 Agent
    print("1. 初始化 KnowledgeQAAgent...")
    agent = KnowledgeQAAgent()
    print("   ✅ Agent 初始化完成")
    
    # 测试查询模式
    print("\n2. 测试查询模式（将触发完整的调用链追踪）...")
    query = "三国演义开头的那首词叫什么名字？"
    print(f"   查询: {query}")
    
    result = agent.chat(query)
    print(f"   回答: {result['answer'][:100]}...")
    print(f"   模式: {result['mode']}")
    print(f"   引用数量: {len(result['sources'])}")
    
    # 测试流式输出
    print("\n3. 测试流式输出（将触发流式追踪）...")
    query2 = "关羽有什么特点？"
    print(f"   查询: {query2}")
    print("   流式回答: ", end="", flush=True)
    
    sources = []
    mode = None
    for chunk in agent.chat_streaming(query2):
        if isinstance(chunk, dict):
            sources = chunk.get("sources", [])
            mode = chunk.get("mode", "unknown")
        else:
            print(chunk, end="", flush=True)
    
    print(f"\n   模式: {mode}")
    print(f"   引用数量: {len(sources)}")
    
    print("\n4. 测试文件上传模式...")
    file_path = "examples/三国演义_部分测试.txt"
    if Path(file_path).exists():
        print(f"   上传文件: {file_path}")
        result_upload = agent.chat("", file_path=file_path)
        print(f"   模式: {result_upload['mode']}")
        print("   ✅ 文件上传完成")
    else:
        print(f"   ⚠️ 测试文件不存在: {file_path}")
    
    print("\n" + "=" * 80)
    print("✅ LangSmith 可观测性测试完成！")
    print("\n📊 查看追踪结果:")
    print("   1. 访问 https://smith.langchain.com/")
    print("   2. 登录您的 LangSmith 账户")
    print("   3. 查看项目: doc_agen_test")
    print("   4. 您将看到完整的调用链追踪信息，包括:")
    print("      - 每个节点的执行时间")
    print("      - 输入和输出数据")
    print("      - 错误信息（如果有）")
    print("      - 性能指标和相似度分数")
    print("=" * 80)


if __name__ == "__main__":
    test_langsmith_tracing()
