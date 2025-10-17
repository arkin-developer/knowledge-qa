"""LangGraph Agent 集成"""

from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pathlib import Path

from .text_processor import TextProcessor
from .llm import LLM
from .memory import MemoryManager
from .file_parser import TextFileParser
from .log_manager import log


class KnowledgeQAState(TypedDict):
    """知识库问答状态"""
    query: str
    file_path: Optional[str]
    mode: str  # "upload", "query"
    context_docs: List[Document]
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


class KnowledgeQAAgent:
    """知识库问答Agent"""

    def __init__(self, text_processor: Optional[TextProcessor] = None,
                 llm: Optional[LLM] = None, memory: Optional[MemoryManager] = None):
        self.text_processor = text_processor or TextProcessor()
        self.llm = llm or LLM()
        self.memory = memory or MemoryManager()

        # 构建LangGraph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        log.info("KnowledgeQAAgent 初始化完成")

    def _build_graph(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(KnowledgeQAState)

        # 添加核心节点
        workflow.add_node("process_file", self._process_file_node)
        workflow.add_node("store_document", self._store_document_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # 设置边连接
        workflow.add_edge("process_file", "store_document")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_error", END)

        # 文档存储后的条件路由
        workflow.add_conditional_edges(
            "store_document",
            self._route_after_store,
            {
                "continue_query": "retrieve_context",
                "end": END
            }
        )

        # 统一入口路由
        def route(state: KnowledgeQAState) -> str:
            if state.get("error"):
                return "handle_error"
            elif state.get("mode") == "upload":
                return "process_file"
            elif state.get("mode") == "query":
                return "retrieve_context"
            else:
                return "handle_error"

        workflow.set_conditional_entry_point(route)

        return workflow

    def _route_after_store(self, state: KnowledgeQAState) -> str:
        """文档存储后的路由决策"""
        return "end"  # 上传模式直接结束

    def _process_file_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """文件处理节点"""
        log.info("执行文件处理")

        try:
            file_path = state.get("file_path")
            if not file_path or not Path(file_path).exists():
                state["error"] = "文件不存在或路径无效"
                return state

            # 文件处理
            log.info(f"开始解析文件: {file_path}")
            text = TextFileParser.parse_file(file_path)
            log.info(f"文件解析完成，文本长度: {len(text)} 字符")

            # 文本分段
            log.info("开始文本分段")
            documents = self.text_processor.split_text(text)
            log.info(f"文本分段完成，共 {len(documents)} 段")

            state["context_docs"] = documents
            log.info("文件处理完成")

        except Exception as e:
            log.error(f"文件处理失败: {e}")
            state["error"] = f"文件处理失败: {str(e)}"

        return state

    def _store_document_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """文档存储节点"""
        log.info("执行文档存储")

        try:
            documents = state.get("context_docs", [])
            if not documents:
                state["error"] = "没有可存储的文档"
                return state

            # 向量化并入库
            log.info("开始向量化并入库")
            self.text_processor.add_documents(documents, batch_size=10)
            log.info("向量化入库完成")

            # 保存向量库
            self.text_processor.save_vector_store()
            log.info("向量库保存完成")

        except Exception as e:
            log.error(f"文档存储失败: {e}")
            state["error"] = f"文档存储失败: {str(e)}"

        return state

    def _retrieve_context_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """上下文检索节点"""
        log.info("执行上下文检索")

        try:
            query = state["query"]
            if not query or not query.strip():
                state["error"] = "查询内容为空"
                return state

            context_docs = self.text_processor.similarity_search(query, k=3)
            state["context_docs"] = context_docs

            log.info(f"检索到 {len(context_docs)} 个相关文档")

        except Exception as e:
            log.error(f"上下文检索失败: {e}")
            state["error"] = f"上下文检索失败: {str(e)}"
            state["context_docs"] = []

        return state

    def _generate_answer_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """答案生成节点"""
        log.info("执行答案生成")

        try:
            query = state["query"]
            context_docs = state["context_docs"]

            # 调用LLM生成回答
            result = self.llm.generate(query, context_docs, use_memory=True)

            state["answer"] = result["answer"]
            state["sources"] = result["sources"]

            # 结果后处理
            answer = state["answer"]
            if len(answer) < 10:
                state["answer"] = "抱歉，我无法提供完整的回答。"

            # 确保引用格式正确
            sources = state["sources"]
            for i, source in enumerate(sources, 1):
                source["index"] = i

            log.info("答案生成完成")

        except Exception as e:
            log.error(f"答案生成失败: {e}")
            state["error"] = f"答案生成失败: {str(e)}"
            state["answer"] = "抱歉，我无法回答这个问题。"
            state["sources"] = []

        return state

    def _handle_error_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """错误处理节点"""
        log.error("执行错误处理")

        error_msg = state.get("error", "未知错误")
        log.error(f"处理错误: {error_msg}")

        # 设置默认错误响应
        state["answer"] = f"处理过程中发生错误: {error_msg}"
        state["sources"] = []

        return state

    def chat(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """对话接口"""
        log.info(f"开始处理用户输入 - 查询: {query}, 文件: {file_path}")

        # 确定模式
        if file_path and Path(file_path).exists():
            mode = "upload"
        else:
            mode = "query"

        # 初始化状态
        initial_state = KnowledgeQAState(
            query=query,
            file_path=file_path,
            mode=mode,
            context_docs=[],
            answer="",
            sources=[],
            error=None
        )

        # 执行图
        final_state = self.app.invoke(initial_state)

        # 返回结果
        result = {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "mode": final_state["mode"]
        }

        log.info("输入处理完成")
        return result

    def chat_streaming(self, query: str, file_path: Optional[str] = None):
        """流式对话接口"""
        log.info(f"开始流式处理用户输入 - 查询: {query}, 文件: {file_path}")

        # 确定模式
        if file_path and Path(file_path).exists():
            mode = "upload"
        else:
            mode = "query"

        # 初始化状态
        state = KnowledgeQAState(
            query=query,
            file_path=file_path,
            mode=mode,
            context_docs=[],
            answer="",
            sources=[],
            error=None
        )

        # 如果有文件，先处理文件
        if mode == "upload":
            # 发送文件处理状态
            yield {"status": "正在处理文件...", "type": "status"}
            state = self._process_file_node(state)
            if not state.get("error"):
                yield {"status": "正在存储文档...", "type": "status"}
                state = self._store_document_node(state)

        # 如果有查询，进行知识检索
        if mode == "query":
            yield {"status": "正在检索相关知识...", "type": "status"}
            state = self._retrieve_context_node(state)

        # 开始生成回答
        yield {"status": "正在生成回答...", "type": "status"}

        # 使用LLM的流式接口
        sources = []
        for chunk in self.llm.generate_streaming(query, state["context_docs"], use_memory=True):
            if isinstance(chunk, dict):
                # 最后的元数据，添加 mode 信息
                chunk["mode"] = mode
                yield chunk
            else:
                # 流式文本内容
                yield chunk

    def clear_memory(self):
        """清空记忆"""
        self.llm.clear_memory()
        log.info("Agent记忆已清空")


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.agent
    print("="*100)
    print("知识库问答Agent - 端到端测试")
    print("="*100)

    print("1. 初始化Agent")
    agent = KnowledgeQAAgent()
    print("   Agent初始化完成\n")

    print("2. 测试文件上传处理")
    file_path = "examples/三国演义_部分测试.txt"
    print(f"   上传文件: {file_path}")
    result_upload = agent.chat("", file_path=file_path)
    print(f"   模式: {result_upload['mode']}\n")

    print("3. 测试纯查询对话")
    query1 = "三国演义开头的那首词叫什么名字？"
    print(f"   Q: {query1}")
    result1 = agent.chat(query1)
    print(f"   A: {result1['answer']}")
    print(f"   模式: {result1['mode']}\n")

    print("4. 测试上下文理解")
    query2 = "它的作者是谁？"
    print(f"   Q: {query2}")
    result2 = agent.chat(query2)
    print(f"   A: {result2['answer']}\n")

    print("5. 测试文件上传后查询")
    query3 = "刘备有什么特点？"
    print(f"   Q: {query3}")
    result3 = agent.chat(query3)
    print(f"   A: {result3['answer']}")
    print(f"   模式: {result3['mode']}\n")

    print("6. 测试流式输出")
    query4 = "关羽有什么特点？"
    print(f"   Q: {query4}")
    print("   A: ", end="", flush=True)

    sources = []
    mode = None
    for chunk in agent.chat_streaming(query4):
        if isinstance(chunk, dict):
            # 最后的元数据
            sources = chunk.get("sources", [])
            mode = chunk.get("mode", "unknown")
            print(f"\n   模式: {mode}")
            print(f"   引用数量: {len(sources)}")
        else:
            # 流式文本内容
            print(chunk, end="", flush=True)
    print("\n")

    print("7. 测试记忆管理")
    print(f"   清空前消息数: {len(agent.llm.memory.get_messages())}")
    agent.clear_memory()
    print(f"   清空后消息数: {len(agent.llm.memory.get_messages())}")

    print("\n✅ Agent测试完成!")
