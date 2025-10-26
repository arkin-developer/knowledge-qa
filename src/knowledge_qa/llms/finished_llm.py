"""判断智能体是否完成大模型回答"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from ..config import settings
from ..log_manager import log


@dataclass
class FinishedState:
    """判断智能体是否完成大模型回答状态"""
    finished: bool
    reason: Optional[str] = None


class FinishedLLM:
    """判断智能体是否完成大模型回答"""

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ).with_structured_output(FinishedState)

        log.info(f"初始化FinishedLLM模型: {settings.llm_model}")

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
你是一个智能助手，用于判断AI智能体是否已经完成了对用户问题的回答。

判断标准：
1. 如果回答完整且准确，返回 finished=True
2. 如果回答不完整、不准确或明确表示无法回答，返回 finished=False
3. 如果回答中提到"找不到相关信息"、"无法回答"等，返回 finished=False
4. 根据判断原因，返回 reason 字段，reason 字段为判断原因的描述；
5. 务必使得返回结构为 FinishedState(finished=bool, reason=str)

请根据用户问题和AI回答的质量来判断是否完成。
        """

    def generate(self, query: str, qa_answer: str) -> FinishedState:
        """根据问题生成回答"""
        try:
            messages: List[BaseMessage] = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=f"用户问题：{query}\n给出答案：{qa_answer}")
            ]
            
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            log.error(f"判断完成状态失败: {e}")
            raise e

if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.llms.finished_llm
    finished_llm = FinishedLLM()
    query = "韩立是如何进入七玄门的？记名弟子初次考验包含哪些关键路段与环节？"
    qa_answer = """根据提供的知识库内容，我无法找到相关信息来回答您的问题。知识库中未提及"韩立"、"七玄门"
或"记名弟子初次考验"的相关内容。"""
    finished_state = finished_llm.generate(query, qa_answer)
    print(finished_state.finished)
    print(finished_state.reason)