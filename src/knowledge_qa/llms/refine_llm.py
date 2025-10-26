""" 验证材料是否跟用户问题相关 """

from dataclasses import dataclass
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.prompts import PromptTemplate

from ..config import settings
from ..log_manager import log


@dataclass
class RefineState:
    """验证材料是否能够完成用户问题"""
    enough: bool  # 是否足够完成用户问题
    reason: Optional[str] = None  # 原因
    suggestions: Optional[str] = None  # 建议二次检索的资料

class RefineLLM:
    """验证材料是否能够完成用户问题"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ).with_structured_output(RefineState)
        log.info(f"初始化RefineLLM模型: {settings.llm_model}")
        
    def get_prompt_template(self) -> str:
        """获取系统提示词"""
        return PromptTemplate.from_template("""
你是一个智能助手，请你根据用户问题和目前掌握的资料判断是否能够完成用户问题，如果不满足你可以根据问题和找到资料精炼出新的问题（交付给reader模型进行二次检索）。
用户问题：{query}
资料：{context}

⚠️注意：
1. 如果用户提供的资料你认为可以回答用户提出的问题，返回 enough=True，reason 字段为空，suggestions 字段为空；
2. 如果用户提供的资料你认为不能回答用户提出的问题，请返回 enough=False，reason 字段为不能回答的原因，suggestions 是你补充的新的问题，用于交付给reader模型进行二次检索；
3. 务必使得返回结构为 RefineState(enough=bool, reason=str, suggestions=str)。

""")

    def generate(self, query: str, context: str) -> RefineState:
        """根据问题生成回答"""
        try:
            input = self.get_prompt_template().format(query=query, context=context)
            return self.llm.invoke(input)
        except Exception as e:
            log.error(f"判断材料是否能够完成用户问题失败: {e}")
            raise e

# 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.llms.refine_llm
if __name__ == "__main__":
    refine_llm = RefineLLM()
    query = "一个目标处于 Grappled（被擒抱/缠住）状态时，会发生什么？列出该状态对目标的具体机械效果。"
    context = "当一个目标处于Grappled（被擒抱/缠住）状态时，会产生以下具体机械效果：\n1. 目标的速度变为0，且不能从任何速度加值中获益（包括法术或其他效果的加速）。\n2. 该状态会在以下情况下结束：\n   - 擒抱者失去行动能力（陷入失能状态）\n   - 有某种效果将被擒抱的目标移出擒抱者的触及范围（比如被雷霆波法术击飞）\n3. 被擒抱的生物可以用一个动作尝试挣脱，需通过力量（运动）或敏捷（体操）检定对抗擒抱者的力量（运动）检定。\n4. 擒抱者移动时可以携带被擒抱的生物，但速度减半（除非目标体型比擒抱者小两级或更多）\n\n这些效果会持续到擒抱状态结束为止。"
    # 错误测试数据
    context = """
### 韩立进入七玄门的方式
1. **亲属引荐**：韩立通过村里在七玄门任职的三叔介绍获得机会（40行）。
2. **集体护送**：与其他孩童一同乘马车前往彩霞山七玄门总部（100行）。

### 记名弟子初次考验关键环节
1. **竹林耐力试炼**：
   - 分散穿越宽广竹林，有师兄监督（126行）。
   - 韩立采取低姿态缓慢前行保存体力。

2. **陡坡攀爬**：
   - 坡度逐渐变陡，需手足并用（131行）。
   - 衣物因摩擦险些破损，考验体能极限。

3. **崖壁竞争**：
   - 正午前攀爬麻绳登顶，舞岩率先完成并嘲讽（147行）。
   - 韩立最终被拉上崖顶（156行）。

4. **资质分流**：
   - 通过者按天赋分核心弟子（七绝堂）与普通弟子，待遇差异显著（171行）。
"""
    result = refine_llm.generate(query, context)
    print(result)