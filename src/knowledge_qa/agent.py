""" 知识库问答agent """

from typing import List, Dict, Any, Optional, TypedDict
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

        # 设置入口点 - 根据mode决定
        workflow.set_entry_point("process_file")

        # 设置条件路由
        workflow.add_conditional_edges(
            "process_file",
            self._file_processing_decision,
            {
                "retrieve": "retrieve_vector_docs",
                "end": END
            }
        )

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
        workflow.add_edge("retrieve_vector_docs", "qa")  # 向量数据库检索节点 -> QA答案生成节点
        workflow.add_edge("qa", "finished")  # QA答案生成节点 -> 完成检查节点
        workflow.add_edge("reader", "refine")  # 阅读本地文档节点 -> 验证材料是否能够完成用户问题节点
        workflow.add_edge("handle_error", END)  # 错误处理节点 -> 结束

        return workflow

    @traceable(name="file_processing_decision")
    def _file_processing_decision(self, state: KnowledgeQAState | Dict[str, Any]) -> str:
        """文件处理决策"""
        if state.get("error"):
            return "end"
        
        mode = state.get("mode", "query")
        if mode == "upload":
            # 上传模式：处理文件后继续检索
            return "retrieve"
        else:
            # 查询模式：直接跳过文件处理，进入检索
            return "retrieve"

    @traceable(name="should_continue")
    def _should_continue(self, state: KnowledgeQAState | Dict[str, Any]) -> str:
        """判断是否需要继续处理"""
        # 检查是否有严重错误（非临时性错误）
        error = state.get("error")
        if error and self._is_critical_error(error):
            log.error(f"严重错误，终止流程: {error}")
            return "end"
        
        # 清除非严重错误，继续处理
        if error and not self._is_critical_error(error):
            log.warning(f"非严重错误，清除后继续: {error}")
            state["error"] = None

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

    @traceable(name="refine_decision")
    def _refine_decision(self, state: KnowledgeQAState | Dict[str, Any]) -> str:
        """refine节点的路由决策"""
        # 检查是否有严重错误
        error = state.get("error")
        if error and self._is_critical_error(error):
            log.error(f"严重错误，终止流程: {error}")
            return "end"
        
        # 清除非严重错误，继续处理
        if error and not self._is_critical_error(error):
            log.warning(f"非严重错误，清除后继续: {error}")
            state["error"] = None

        refine_state = state.get("refine_state")
        if refine_state:
            # 处理字典格式的状态
            if isinstance(refine_state, dict):
                enough = refine_state.get("enough", False)
            else:
                enough = refine_state.enough

            if enough:
                log.info("材料验证通过，回到QA节点重新生成答案")
                return "back_to_qa"
            else:
                log.info("材料不足，继续reader节点获取更多材料")
                return "continue_reader"
        else:
            log.warning("没有refine_state，结束流程")
            return "end"
    
    def _is_critical_error(self, error: str) -> bool:
        """判断是否为严重错误"""
        critical_errors = [
            "文件不存在或路径无效",
            "查询内容为空",
            "没有可分段的文档"
        ]
        return any(critical in error for critical in critical_errors)

    @traceable(name="chat")
    def chat(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """对话接口"""
        result = self.app.invoke({
            "query": query,
            "mode": "upload" if file_path else "query",
            "file_path": file_path if file_path else None
        })
        # 保存最后的状态用于调试
        self._last_state = result
        return result
    
    def get_last_state(self) -> Dict[str, Any]:
        """获取最后一次执行的状态"""
        return getattr(self, '_last_state', {})

    @traceable(name="process_file_node")
    def _process_file_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """文件处理节点"""
        log.info("执行文件处理")

        try:
            mode = state.get("mode", "query")
            
            if mode == "upload":
                # 上传模式：处理文件
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
            else:
                # 查询模式：跳过文件处理
                log.info("查询模式，跳过文件处理")
                
        except Exception as e:
            log.error(f"文件处理失败: {e}")
            state["error"] = f"文件处理失败: {str(e)}"

        return state

    @traceable(name="retrieve_vector_docs_node")
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

    @traceable(name="qa_node")
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
                            metadata={"filename": fragment.filename,
                                      "start_line": fragment.start_line, "end_line": fragment.end_line}
                        ))
                    else:
                        # 处理字典格式
                        context_docs.append(Document(
                            page_content=fragment.get('content', ''),
                            metadata={"filename": fragment.get('filename', ''), "start_line": fragment.get(
                                'start_line', 0), "end_line": fragment.get('end_line', 0)}
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

    @traceable(name="finished_node")
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

    @traceable(name="reader_node")
    def _reader_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """阅读本地文档节点"""
        log.info("执行阅读本地文档")

        try:
            query = state.get("query")
            suggestions = state.get("suggestions")
            # 如果存在改进建议，证明不是第一次查询资料
            if suggestions:
                log.info(f"存在改进建议，使用改进建议进行二次检索: {suggestions}")
                query = suggestions
   
            # 重试机制：如果查不到内容，允许重试两次
            max_retries = 3
            fragments = []
            
            for attempt in range(max_retries):
                try:
                    self.reader_llm.clear_fragments_meta()  # 清空文档片段元数据列表
                    _ = self.reader_llm.generate(query)  # 等待执行补充文档片段
                    self.reader_llm.update_fragments()  # 更新文档片段列表,获取完整文档片段
                    fragments = self.reader_llm.get_fragments() # 获取完整文档片段
                    log.info(f"fragments: {fragments}")
                    
                    if fragments and len(fragments) > 0:
                        log.info(f"第 {attempt + 1} 次尝试成功获取 {len(fragments)} 个文档片段")
                        break
                    else:
                        log.warning(f"第 {attempt + 1} 次尝试未找到文档片段")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)  # 等待1秒后重试
                        
                except Exception as retry_e:
                    log.warning(f"第 {attempt + 1} 次尝试失败: {retry_e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        raise retry_e
            
            if not fragments or len(fragments) == 0:
                log.warning("多次尝试后仍未找到文档片段，继续处理")
                fragments = []
                
            if state.get("document_fragments") is None:
                state["document_fragments"] = []
            # 追加文档
            state["document_fragments"].extend(fragments)  
            log.info(f"document_fragments: 目前文档数量: {len(state.get('document_fragments'))}")
        except Exception as e:
            log.error(f"阅读本地文档失败: {e}")
            state["error"] = f"阅读本地文档失败: {str(e)}"
            state["document_fragments"] = []
        return state

    @traceable(name="refine_node")
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
            log.info(f"refine_state: {refine_state}")
            # 处理字典格式的状态
            if isinstance(refine_state, dict):
                enough = refine_state.get("enough", False)
                suggestions = refine_state.get("suggestions", "")
                state["refine_suggestions"] = suggestions
                log.info(f"材料验证结果: enough={enough}, suggestions={suggestions}")
            else:
                enough = refine_state.enough
                suggestions = refine_state.suggestions
                state["refine_suggestions"] = suggestions
                log.info(f"材料验证结果: enough={enough}, suggestions={suggestions}")
        except Exception as e:
            log.error(f"验证材料是否能够完成用户问题失败: {e}")
            state["error"] = f"验证材料是否能够完成用户问题失败: {str(e)}"
            state["refine_suggestions"] = "验证材料是否能够完成用户问题失败"

        return state

    @traceable(name="handle_error_node")
    def _handle_error_node(self, state: KnowledgeQAState | Dict[str, Any]) -> KnowledgeQAState:
        """错误处理节点"""
        log.error(f"处理错误: {state.get('error')}")
        return state


# 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.agent
if __name__ == "__main__":
    kb_agent = KnowledgeQAAgent()
    
    # 测试查询模式

    # 测试上传模式（如果有文件的话）
    # print("=== 上传模式测试 ===")
    # state = agent.app.invoke({
    #     "query": "测试查询",
    #     "mode": "upload",
    #     "file_path": "examples/Player's Handbook.md"
    # })
    # print("上传结果:", state.get("qa_answer", "无答案"))

    # 测试RAG 10题
    # 凡人修仙传
    # result = kb_agent.chat("韩立是如何进入七玄门的？记名弟子初次考验包含哪些关键路段与环节？")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()

    # result = kb_agent.chat("墨大夫的真实来历与核心目的是什么？他与韩立关系的转折点发生在哪些事件上？")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()

    # result = kb_agent.chat("神手谷中的神秘小瓶具备什么规律与用途？韩立如何验证并应用？")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()

    # result = kb_agent.chat("落日峰死契血斗前后的关键人物与转折是什么？韩立如何扭转战局？")    
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()

    # result = kb_agent.chat("韩立已系统掌握的法术与其局限是什么？他如何法武并用？")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()
    
    # # 操作文档
    # result = kb_agent.chat("人类（Human）的标准种族特性里，能力值（Ability Scores）如何改变？人类还会获得哪些语言？")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()
    
    # result = kb_agent.chat("在战斗中当你使用一次动作（Action）时，下面哪项不是标准动作？（A）Dash（冲刺） （B）Disengage（脱离） （C）Dodge（躲闪） （D）Teleport（瞬移）")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()
    
    result = kb_agent.chat("如果你在某回合用 bonus action（奖励动作）施放了一个法术，你还能在同一回合再施放一个非戏法（cantrip）的法术吗？为什么？")
    print("query: \n", kb_agent.get_last_state().get("query"))
    print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    kb_agent.qa_llm.clear_memory()
    
    # result = kb_agent.chat("简述短休息（Short Rest）与长休息（Long Rest）的主要区别与效果（至少包含各自持续时间与恢复内容）。")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()
    
    # result = kb_agent.chat("当一个目标处于 Grappled（被擒抱/缠住）状态时，会发生什么？列出该状态对目标的具体机械效果。")
    # print("query: \n", kb_agent.get_last_state().get("query"))
    # print("answer: \n", kb_agent.get_last_state().get("qa_answer", "无答案"))
    # print("fragments: \n", kb_agent.get_last_state().get("document_fragments", "无文档片段"))
    # kb_agent.qa_llm.clear_memory()
