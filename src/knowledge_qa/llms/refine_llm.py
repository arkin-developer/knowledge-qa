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
你是一个严格的质量评估助手，负责判断检索到的资料是否能够充分回答用户的问题。
如果不满足你可以根据问题和找到资料精炼出新的问题，交付给reader模型进行二次检索。

用户问题：{query}
资料：{context}

## 严格评估标准

### 满足条件（enough=True）：
1. **内容完整性**：资料必须包含回答用户问题所需的所有关键信息
2. **信息准确性**：资料中的信息必须准确、可靠，没有明显错误
3. **相关性**：资料必须与用户问题高度相关，不能是泛泛而谈
4. **具体性**：资料必须提供具体的细节、数据或实例，不能只是概念性描述
5. **可操作性**：如果问题需要具体操作，资料必须包含足够的操作细节

### 不满足条件（enough=False）：
1. **信息不足**：资料缺少关键信息，无法完整回答问题
2. **过于宽泛**：资料太笼统，缺乏具体细节
3. **相关性差**：资料与问题关联度不高
4. **信息过时**：资料可能已经过时或不准确
5. **缺乏实例**：需要具体例子时，资料没有提供

## 评估要求
- 必须严格按照上述标准进行评估
- 宁可严格也不要宽松
- 如果资料有任何不足，必须返回 enough=False, 必须提供具体的改进建议

## 重要要求
- **reason 字段必须填写**：无论 enough 是 True 还是 False，都必须详细说明判断原因
- **suggestions**: 如果 enough=False，必须结合资料和问题精炼出缺失的关键信息（可以换个角度提问），作为新的问题交付给reader模型进行二次检索。

返回格式：RefineState(enough=bool, reason=str, suggestions=str)
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