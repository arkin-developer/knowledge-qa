"""上下文管理"""

from typing import Optional
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import BaseMessage

from .config import settings
from .log_manager import log


class MemoryManager:
    """记忆管理器，基于 LangChain Memory"""
    
    def __init__(self, window_size: Optional[int] = None, return_messages: bool = True):
        window_size = window_size or settings.memory_window_size
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=return_messages,
            memory_key="chat_history"
        )
        log.info(f"初始化记忆管理器，窗口大小: {window_size}")
        
    def add_exchange(self, user_input: str, ai_response: str) -> None:
        """添加一轮对话"""
        self.memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        log.debug(f"保存对话 - 用户: {user_input[:50]}... AI: {ai_response[:50]}...")
    
    def get_history(self) -> dict:
        """获取历史对话"""
        return self.memory.load_memory_variables({})
    
    def get_messages(self) -> list[BaseMessage]:
        """获取消息列表"""
        history = self.get_history()
        return history.get("chat_history", [])
    
    def clear(self) -> None:
        """清空历史记录"""
        self.memory.clear()
        log.info("已清空历史记录")
    
    def get_memory_key(self) -> str:
        """获取记忆键"""
        return self.memory.memory_key


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.memory
    manager = MemoryManager(window_size=30)
    
    manager.add_exchange("你好", "你好！有什么可以帮你的？")
    manager.add_exchange("今天天气怎么样？", "抱歉，我无法获取实时天气信息。")
    manager.add_exchange("你能做什么？", "我可以回答问题、提供信息等。")
    manager.add_exchange("谢谢", "不客气！")
    
    print("历史对话（窗口=3，只保留最近30轮）:")
    history = manager.get_history()
    print(history)
    
    print("\n消息列表:")
    messages = manager.get_messages()
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")
