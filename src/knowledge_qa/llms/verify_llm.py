"""答案质量评估LLM"""

from dataclasses import dataclass
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from ..config import settings
from ..log_manager import log


@dataclass
class VerifyState:
    """答案质量评估状态"""
    satisfied: bool
    reason: Optional[str] = None
    suggestions: Optional[str] = None 


class VerifyLLM:
    """答案质量评估LLM"""

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ).with_structured_output(VerifyState)
        log.info(f"初始化VerifyLLM模型: {settings.llm_model}")

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
你是一个智能助手，请你根据用户问题，用户第一轮问答的结果，再根据用户二次查询的资料判断是否满足用户的回答。

评估标准：
1. 如果用户提供的资料你认为可以回答用户提出的问题，返回 satisfied=True，reason 字段为空，suggestions 字段为空；
3. 如果用户提供的资料你认为不能回答用户提出的问题，请返回 reason 字段，reason 字段为不能回答的原因，suggestions 字段为改进建议，建议二次检索需要注意的地方；
5. 务必使得返回结构为 VerifyState(satisfied=bool, reason=str, suggestions=str)。
        """

    def generate(self, query: str, answer: str, context: str) -> VerifyState:
        """根据问题生成回答"""
        user_prompt = f"问题：{query}\n用户答案：{answer}\n资料：{context}"

        try:
            messages: List[BaseMessage] = [
                SystemMessage(content=self.get_system_prompt())
            ]
            messages.append(HumanMessage(content=user_prompt))
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            log.error(f"判断资料是否满足失败: {e}")
            raise e


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.llms.verify_llm
    verify_llm = VerifyLLM()
    query = "韩立是如何进入七玄门的？记名弟子初次考验包含哪些关键路段与环节？"
    answer = "根据提供的知识库内容，我无法找到相关信息来回答您的问题。知识库中未提及"
    context = "？"
    verify_state = verify_llm.generate(query, answer, context)
    print(verify_state)