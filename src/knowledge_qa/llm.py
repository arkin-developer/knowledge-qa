"""大模型-提示词工程"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langsmith import traceable

from .config import settings
from .log_manager import log
from .memory import MemoryManager


class LLM:
    """LLM模型"""

    def __init__(self, memory: Optional[MemoryManager] = None):
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        )
        self.memory = memory or MemoryManager()
        log.info(f"初始化LLM模型: {settings.llm_model}")

    def get_system_prompt(self) -> str:
        """获取提示词"""
        system_prompt = f"""
你是一个知识库问答助手，你只能基于提供的知识库内容回答用户问题。
严格遵循以下规则：
1. 只能使用提供的知识库内容来回答问题，不得使用自身知识或外部信息；
2. 如果知识库中没有相关信息，请明确告知"根据提供的知识库内容，我无法找到相关信息来回答您的问题"；
3. 回答时必须明确引用来源，使用[1]、[2]等规范格式标注知识库中的具体内容；
4. 保持回答的准确性和简洁性，确保所有信息都来自知识库；
5. 不得编造、推测或添加知识库中不存在的信息。
        """
        return system_prompt

    @traceable(name="llm_generate")
    def generate(self, query: str, context_docs: List[Any], use_memory: bool = True) -> Dict[str, Any]:
        """根据知识库上下文回答问题"""
        try:
            import time
            start_time = time.time()

            context_text = "\n\n".join(
                [f"{i+1}. {doc.page_content}" for i, doc in enumerate(context_docs)])

            user_prompt = f"""
上下文信息：
{context_text}

用户问题：{query}
            """

            messages: List[BaseMessage] = [
                SystemMessage(content=self.get_system_prompt())]

            if use_memory:
                history_messages = self.memory.get_messages()
                if history_messages:
                    messages.extend(history_messages)
                    log.debug(f"加载历史对话: {len(history_messages)} 条消息")

            messages.append(HumanMessage(content=user_prompt))

            response = self.llm.invoke(messages)
            end_time = time.time()
            response_time = end_time - start_time
            log.info(f"回答问题耗时: {response_time:.2f}秒")

            if use_memory:
                self.memory.add_exchange(query, response.content)
                log.debug("已保存本轮对话到记忆")

            sources = []
            for i, doc in enumerate(context_docs):
                source_info = {
                    "index": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)

            result = {
                "answer": response.content,
                "sources": sources
            }
            return result
        except Exception as e:
            log.error(f"回答问题失败: {e}")
            raise e

    @traceable(name="llm_generate_streaming")
    def generate_streaming(self, query: str, context_docs: List[Any], use_memory: bool = True):
        """根据知识库上下文流式回答问题"""
        try:
            import time
            start_time = time.time()

            context_text = "\n\n".join(
                [f"{i+1}. {doc.page_content}" for i, doc in enumerate(context_docs)])

            user_prompt = f"""
上下文信息：
{context_text}

用户问题：{query}
            """

            messages: List[BaseMessage] = [
                SystemMessage(content=self.get_system_prompt())]

            if use_memory:
                history_messages = self.memory.get_messages()
                if history_messages:
                    messages.extend(history_messages)
                    log.debug(f"加载历史对话: {len(history_messages)} 条消息")

            messages.append(HumanMessage(content=user_prompt))

            full_response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            end_time = time.time()
            response_time = end_time - start_time
            log.info(f"流式回答问题耗时: {response_time:.2f}秒")

            if use_memory:
                self.memory.add_exchange(query, full_response)
                log.debug("已保存本轮对话到记忆")

            sources = []
            for i, doc in enumerate(context_docs):
                source_info = {
                    "index": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)

            yield {
                "sources": sources,
                "full_response": full_response
            }

        except Exception as e:
            log.error(f"流式回答问题失败: {e}")
            raise e

    def clear_memory(self) -> None:
        """清空对话记忆"""
        self.memory.clear()


if __name__ == "__main__":
    from .file_parser import TextFileParser
    from .text_processor import TextProcessor
    from .vector_store import VectorStore

    print("知识库问答系统 - 端到端测试\n")

    print("1. 加载文档并构建向量库")
    text = TextFileParser.parse_file("examples/三国演义.txt")
    print(f"   文档长度: {len(text)} 字符")

    processor = TextProcessor()
    vector_store = VectorStore()
    documents = processor.split_text(text)
    vector_store.add_documents(documents[:20], batch_size=10)
    print(f"   向量库构建完成，共 {len(documents[:20])} 段\n")

    llm = LLM()

    print("2. 基础问答测试")
    query1 = "三国演义开头的那首词叫什么名字？"
    print(f"   Q: {query1}")
    result1 = llm.generate(query1, vector_store.similarity_search(query1, k=3))
    print(f"   A: {result1['answer']}\n")

    print("3. 上下文理解测试")
    query2 = "它的作者是谁？"
    print(f"   Q: {query2}")
    result2 = llm.generate(query2, vector_store.similarity_search("三国演义作者", k=3))
    print(f"   A: {result2['answer']}")
    print(f"   对话历史: {len(llm.memory.get_messages())} 条消息\n")

    print("4. 流式输出测试")
    query3 = "刘备有什么特点？"
    print(f"   Q: {query3}")
    print("   A: ", end="", flush=True)
    for chunk in llm.generate_streaming(query3, vector_store.similarity_search("刘备", k=3)):
        if not isinstance(chunk, dict):
            print(chunk, end="", flush=True)
    print("\n")

    print("5. 记忆管理测试")
    print(f"   清空前: {len(llm.memory.get_messages())} 条消息")
    llm.clear_memory()
    print(f"   清空后: {len(llm.memory.get_messages())} 条消息")

    print("\n✅ 测试完成!")
