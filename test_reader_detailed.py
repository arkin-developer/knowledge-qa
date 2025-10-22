#!/usr/bin/env python3
"""详细测试ReaderLLM"""

import sys
import os
sys.path.append('src')

from knowledge_qa.llms.reader_llm import ReaderLLM
from langchain_core.messages import SystemMessage, HumanMessage

def test_reader_detailed():
    print("详细测试ReaderLLM")
    
    # 初始化ReaderLLM
    reader_llm = ReaderLLM()
    
    # 直接测试LLM调用
    print("1. 测试LLM直接调用:")
    try:
        messages = [
            SystemMessage(content=reader_llm.get_system_prompt()),
            HumanMessage(content="用户的问题：韩立是如何进入七玄门的？")
        ]
        
        response = reader_llm.llm.invoke(messages)
        print(f"响应类型: {type(response)}")
        print(f"响应内容: {response}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reader_detailed()
