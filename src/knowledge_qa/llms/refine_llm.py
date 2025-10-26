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

## 评估标准

### 满足条件（enough=True）：
1. **内容完整性**：资料包含回答用户问题的主要信息
2. **相关性**：资料与用户问题相关，能够提供有价值的回答
3. **具体性**：资料提供了一定的具体细节或实例
4. **可回答性**：基于现有资料能够给出合理的回答

### 不满足条件（enough=False）：
1. **信息严重不足**：资料完全无法回答用户问题
2. **完全无关**：资料与问题完全不相关
3. **信息错误**：资料包含明显错误信息
4. **过于模糊**：资料过于模糊，无法提供任何有用信息

## 评估要求
- 基于现有资料是否能够给出合理回答进行判断
- 如果资料能够回答用户问题的主要部分，应返回 enough=True
- 只有在资料完全无法回答问题时，才返回 enough=False

## 重要要求
- **reason 字段必须填写**：无论 enough 是 True 还是 False，都必须详细说明判断原因
- **suggestions**: 如果 enough=False，必须提供新的搜索策略：
  - **分析原因**：基于已查到的资料，分析为什么资料不足（缺少哪些关键信息）
  - **提取新关键字**：从已查到的资料中寻找相关但不同的关键词、人名、地名、概念等
  - **换个角度**：基于资料内容，建议从不同角度重新搜索（如同义词、相关概念、具体细节等）
  - **搜索策略**：指导reader模型使用从现有资料中提取的新关键词进行搜索

## 关键字建议原则
- **基于现有资料**：仔细分析已查到的资料，从中提取新的关键词
- **避免重复**：不使用已经问题中已经使用过的关键词
- **提取新信息**：从资料中找出人名、地名、概念、事件等新关键词
- **相关但不同**：提供与原始问题相关但角度不同的搜索词
- **具体化搜索**：基于资料内容，建议更具体或更宽泛的搜索词
- 示例：
  * 如果搜索"韩立 七玄门"只找到入门信息，从资料中发现"三叔"、"记名弟子"、"炼骨崖"等关键词，建议搜索"三叔 推荐"、"记名弟子 考验"、"炼骨崖 测试"等
  * 如果搜索"法术 修炼"只找到基础信息，从资料中发现"口诀"、"墨大夫"、"内力"等关键词，建议搜索"口诀 修炼"、"墨大夫 传授"、"内力 运行"等
  * 如果搜索"死契 血斗"只找到部分信息，从资料中发现"落日峰"、"王绝楚"、"贾天龙"等关键词，建议搜索"落日峰 决斗"、"王绝楚 贾天龙"、"金光上人"等

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