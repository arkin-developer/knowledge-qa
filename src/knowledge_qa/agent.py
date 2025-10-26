""" 知识库问答agent """

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pathlib import Path
from langsmith import traceable

from .text_processor import TextProcessor
from .file_parser import FileParser
from .vector_store import VectorStore
from .llms.reader_llm import ReaderLLM, DocumentFragment
from .llms.qa_llm import QALLM
from .llms.finished_llm import FinishedLLM, FinishedState
from .llms.refine_llm import RefineLLM, RefineState
from .log_manager import log
from .config import settings


# 知识库问答状态
class KnowledgeQAState(TypedDict):
    """知识库问答状态"""
    query: str  # 用户的原始问题
    vector_docs: str  # 向量数据库的检索结果
    mode: str  # "upload", "query" 上传文件模式（文本文件嵌入）或 查询模式
    refine_state: Optional[RefineState]  # 验证材料是否能够完成用户问题
    document_fragments: Optional[List[DocumentFragment]]  # 通过本地文档查询的文档片段
    qa_answer: str  # QA大模型回答
    finished_state: Optional[FinishedState]  # 判断 query 是否回答完成
    refine_suggestions: Optional[str]  # 验证模型给出的建议，可以传入reader模型进行二次检索
    error: Optional[str]  # 错误信息，如果发生错误，则需要处理错误

# 知识库问答Agent


class KnowledgeQAAgent:
    """知识库问答Agent"""

    def __init__(self):
        self.text_processor = TextProcessor()  # 文本处理器
        self.vector_store = VectorStore()  # 向量存储
        self.qa_llm = QALLM()  # 根据上下文知识回答模型
        self.reader_llm = ReaderLLM()  # 阅读本地文档的模型
        self.finished_llm = FinishedLLM()  # 判断 query 是否回答完成模型
        self.refine_llm = RefineLLM()  # 验证材料是否能够完成用户问题模型

        # 构建LangGraph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        log.info("KnowledgeQAAgent 初始化完成")

    def _build_graph(self) -> StateGraph:
        """构建LangGraph"""
        workflow = StateGraph(KnowledgeQAState)

        # 添加核心节点
        workflow.add_node("process_file", self._process_file_node)  # 文件处理节点
        workflow.add_node("retrieve_vector_docs",
                          self._retrieve_vector_docs_node)  # 向量数据库检索节点
        workflow.add_node("qa", self._qa_node)  # QA答案生成节点
        workflow.add_node("finished", self._finished_node)  # 完成检查节点
        workflow.add_node("reader", self._reader_node)  # 阅读本地文档节点
        workflow.add_node("refine", self._refine_node)  # 验证材料是否能够完成用户问题节点
        workflow.add_node("handle_error", self._handle_error_node)  # 错误处理节点

        # 设置入口点
        workflow.set_entry_point("retrieve_vector_docs")

        # 设置条件路由
        workflow.add_conditional_edges(
            "finished",
            self._should_continue,
            {
                "continue": "reader",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "refine",
            self._refine_decision,
            {
                "back_to_qa": "qa",
                "continue_reader": "reader",
                "end": END
            }
        )

        # 静态边连接
        # 向量数据库检索节点 -> QA答案生成节点
        workflow.add_edge("retrieve_vector_docs", "qa")
        workflow.add_edge("qa", "finished")  # QA答案生成节点 -> 完成检查节点
        workflow.add_edge("reader", "refine")  # 阅读本地文档节点 -> 验证材料是否能够完成用户问题节点
        workflow.add_edge("handle_error", END)  # 错误处理节点 -> 结束

        return workflow

    def _should_continue(self, state: KnowledgeQAState | Dict[str, Any]) -> str:
        """判断是否需要继续处理"""
        if state.get("error"):
            return "end"

        finished_state = state.get("finished_state")
        if finished_state:
            # 处理字典格式的状态
            if isinstance(finished_state, dict):
                finished = finished_state.get("finished", False)
            else:
                finished = finished_state.finished
            
            if finished:
                return "end"
        return "continue"

    def _refine_decision(self, state: KnowledgeQAState | Dict[str, Any]) -> str:
        """refine节点的路由决策"""
        if state.get("error"):
            return "end"

        refine_state = state.get("refine_state")
        if refine_state:
            # 处理字典格式的状态
            if isinstance(refine_state, dict):
                enough = refine_state.get("enough", False)
            else:
                enough = refine_state.enough
            
            if enough:
                return "back_to_qa"
            else:
                return "continue_reader"
        return "end"

    def _process_file_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """文件处理节点"""
        log.info("执行文件处理")

        try:
            file_path = state.get("file_path")
            if not file_path or not Path(file_path).exists():
                state["error"] = "文件不存在或路径无效"
                return state
            text = FileParser.parse_file(file_path)
            documents = self.text_processor.split_text(text)
            if not documents:
                state["error"] = "没有可分段的文档"
                return state
            self.vector_store.add_documents(documents, batch_size=20)
            self.vector_store.save_vector_store()
            log.info("文件处理完成")
        except Exception as e:
            log.error(f"文件处理失败: {e}")
            state["error"] = f"文件处理失败: {str(e)}"

    def _retrieve_vector_docs_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """向量数据库检索节点"""
        log.info("执行上下文检索")

        try:
            query = state.get("query")
            if not query or not query.strip():
                state["error"] = "查询内容为空"
                return state
            vector_docs = self.vector_store.similarity_search(
                query, k=settings.search_k)
            state["vector_docs"] = vector_docs
            log.info(f"检索到 {len(vector_docs)} 个相关文档")
        except Exception as e:
            log.error(f"上下文检索失败: {e}")
            state["error"] = f"上下文检索失败: {str(e)}"
            state["vector_docs"] = []

        return state

    def _qa_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """QA答案生成节点"""
        log.info("执行答案生成")

        try:
            query = state.get("query")
            vector_docs = state.get("vector_docs")
            if not query or not query.strip():
                state["error"] = "查询内容为空"
                return state
            if state.get("document_fragments") and len(state.get("document_fragments")) > 0:
                # 将DocumentFragment对象转换为Document对象
                from langchain_core.documents import Document
                context_docs = []
                for fragment in state.get("document_fragments"):
                    if hasattr(fragment, 'content'):
                        context_docs.append(Document(
                            page_content=fragment.content,
                            metadata={"filename": fragment.filename, "start_line": fragment.start_line, "end_line": fragment.end_line}
                        ))
                    else:
                        # 处理字典格式
                        context_docs.append(Document(
                            page_content=fragment.get('content', ''),
                            metadata={"filename": fragment.get('filename', ''), "start_line": fragment.get('start_line', 0), "end_line": fragment.get('end_line', 0)}
                        ))
            else:
                context_docs = vector_docs
            answer = self.qa_llm.generate(query, context_docs)
            state["qa_answer"] = answer["answer"]
        except Exception as e:
            log.error(f"QA答案生成失败: {e}")
            state["error"] = f"QA答案生成失败: {str(e)}"
            state["qa_answer"] = FinishedState(
                finished=False, reason="QA答案生成失败")

        return state

    def _finished_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """完成检查节点"""
        log.info("执行完成检查")

        try:
            query = state.get("query")
            qa_answer = state.get("qa_answer")
            # 处理qa_answer可能是FinishedState对象的情况
            if isinstance(qa_answer, FinishedState):
                qa_answer_text = qa_answer.reason or ""
            else:
                qa_answer_text = str(qa_answer) if qa_answer else ""
            
            if not query or not query.strip() or not qa_answer_text or not qa_answer_text.strip():
                state["error"] = "查询内容为空"
                return state
            finished_state = self.finished_llm.generate(query, qa_answer_text)
            state["finished_state"] = finished_state
        except Exception as e:
            log.error(f"完成检查失败: {e}")
            state["error"] = f"完成检查失败: {str(e)}"
            state["finished_state"] = FinishedState(
                finished=False, reason="完成检查失败")

        return state

    def _reader_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """阅读本地文档节点"""
        log.info("执行阅读本地文档")

        try:
            query = state.get("query")
            if not query or not query.strip():
                state["error"] = "查询内容为空"
                return state
            self.reader_llm.clear_fragments_meta()  # 清空文档片段元数据列表
            _ = self.reader_llm.generate(query)  # 等待执行补充文档片段
            self.reader_llm.update_fragments()  # 更新文档片段列表,获取完整文档片段
            fragments = self.reader_llm.get_fragments()
            if not fragments or len(fragments) == 0:
                state["error"] = "没有找到相关文档片段"
                return state
            if state.get("document_fragments") is None:
                state["document_fragments"] = []
            state["document_fragments"].extend(
                fragments)  # 这里采取追加的方式，因为可能多次查询，多次追加
        except Exception as e:
            log.error(f"阅读本地文档失败: {e}")
            state["error"] = f"阅读本地文档失败: {str(e)}"
            state["document_fragments"] = []
        return state

    def _refine_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """验证模型给出的建议节点"""
        log.info("执行验证材料是否能够完成用户问题")

        try:
            query = state.get("query")
            document_fragments = state.get("document_fragments")
            if not query or not query.strip() or not document_fragments or len(document_fragments) == 0:
                state["error"] = "查询内容为空"
                return state
            # 将DocumentFragment对象转换为字符串格式
            context_text = ""
            for fragment in document_fragments:
                if hasattr(fragment, 'content'):
                    context_text += f"文件: {fragment.filename}\n行号: {fragment.start_line}-{fragment.end_line}\n内容: {fragment.content}\n\n"
                else:
                    # 处理字典格式
                    context_text += f"文件: {fragment.get('filename', '')}\n行号: {fragment.get('start_line', '')}-{fragment.get('end_line', '')}\n内容: {fragment.get('content', '')}\n\n"
            
            refine_state = self.refine_llm.generate(query, context_text)
            state["refine_state"] = refine_state
            # 处理字典格式的状态
            if isinstance(refine_state, dict):
                state["refine_suggestions"] = refine_state.get("suggestions", "")
            else:
                state["refine_suggestions"] = refine_state.suggestions
        except Exception as e:
            log.error(f"验证材料是否能够完成用户问题失败: {e}")
            state["error"] = f"验证材料是否能够完成用户问题失败: {str(e)}"
            state["refine_suggestions"] = "验证材料是否能够完成用户问题失败"

        return state

    def _handle_error_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """错误处理节点"""
        log.error(f"处理错误: {state.get('error')}")
        return state


# 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.agent
if __name__ == "__main__":
    agent = KnowledgeQAAgent()
    state = agent.app.invoke({
        "query": "一个目标处于 Grappled（被擒抱/缠住）状态时，会发生什么？列出该状态对目标的具体机械效果。"
    })
    print(state)

