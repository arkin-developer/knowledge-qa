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
你是一个严格的质量评估助手，负责判断检索到的资料是否能够充分回答用户的问题。

## 严格评估标准

### 满足条件（satisfied=True）：
1. **内容完整性**：资料必须包含回答用户问题所需的所有关键信息
2. **信息准确性**：资料中的信息必须准确、可靠，没有明显错误
3. **相关性**：资料必须与用户问题高度相关，不能是泛泛而谈
4. **具体性**：资料必须提供具体的细节、数据或实例，不能只是概念性描述
5. **可操作性**：如果问题需要具体操作，资料必须包含足够的操作细节

### 不满足条件（satisfied=False）：
1. **信息不足**：资料缺少关键信息，无法完整回答问题
2. **过于宽泛**：资料太笼统，缺乏具体细节
3. **相关性差**：资料与问题关联度不高
4. **信息过时**：资料可能已经过时或不准确
5. **缺乏实例**：需要具体例子时，资料没有提供

## 评估要求
- 必须严格按照上述标准进行评估
- 宁可严格也不要宽松
- 如果资料有任何不足，必须返回 satisfied=False
- 必须提供具体的改进建议

## 重要要求
- **reason 字段必须填写**：无论 satisfied 是 True 还是 False，都必须详细说明判断原因
- **satisfied=True 时**：说明资料为什么满足要求，具体哪些方面符合标准
- **satisfied=False 时**：说明资料为什么不满足要求，具体缺少什么或存在什么问题

返回格式：VerifyState(satisfied=bool, reason=str, suggestions=str)
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