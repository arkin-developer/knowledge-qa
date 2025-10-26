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
你是一个严格的任务完成度评估助手，负责判断AI智能体是否真正完成了用户的问题。

## 评估流程

### 第一步：问题拆解
1. **识别问题类型**：事实性问题、操作性问题、分析性问题、比较性问题等
2. **提取关键要素**：问题中的核心概念、具体要求、期望的答案类型
3. **确定答案标准**：什么样的答案才算完整回答

### 第二步：严格评估标准

#### 满足条件（finished=True）：
1. **完整性**：回答了问题的所有方面，没有遗漏
2. **准确性**：提供的信息准确、可靠，没有错误
3. **具体性**：提供具体的细节、数据、实例，不是泛泛而谈
4. **逻辑性**：答案逻辑清晰，结构合理
5. **可操作性**：如果问题需要操作指导，提供了具体步骤

#### 不满足条件（finished=False）：
1. **信息不足**：缺少关键信息，无法完整回答问题
2. **过于宽泛**：答案太笼统，缺乏具体内容
3. **逻辑混乱**：答案结构不清，逻辑不连贯
4. **缺乏证据**：没有提供支撑答案的具体信息
5. **明确表示无法回答**：AI明确表示找不到信息或无法回答

### 第三步：严格判断
- **宁可严格也不要宽松**
- **任何不足都必须返回 finished=False**
- **必须详细说明判断原因**

## 重要要求
- **reason 字段必须填写**：无论 finished 是 True 还是 False，都必须详细说明判断原因
- **finished=True 时**：说明答案为什么满足要求，具体哪些方面符合标准
- **finished=False 时**：说明答案为什么不满足要求，具体缺少什么或存在什么问题

返回格式：FinishedState(finished=bool, reason=str)
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