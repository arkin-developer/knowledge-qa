"""LangGraph Agent 集成"""

from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pathlib import Path
from langsmith import traceable

from .text_processor import TextProcessor
from .vector_store import VectorStore
from .llms.reader_llm import ReaderLLM, ReaderResult, DocumentFragment
from .llms.qa_llm import QALLM
from .llms.finished_llm import FinishedLLM, FinishedState
from .llms.verify_llm import VerifyLLM, VerifyState
from .file_parser import FileParser
from .log_manager import log
from .config import settings


class KnowledgeQAState(TypedDict):
    """知识库问答状态"""
    query: str # 用户的原始问题
    file_path: Optional[str] # 文件路径
    mode: str  # "upload", "query"
    context_docs: List[Document] # 上下文文档(向量数据库的检索结果)
    reader_result: Optional[ReaderResult] # 通过本地文档查询的文档片段
    qa_answer: str # QA大模型的回答
    verify_state: Optional[VerifyState] # 通过reader模型查询的资料是否满足用户问题的验证状态
    finished_state: Optional[FinishedState] # 判断智能体是否完成大模型回答的状态
    suggestions: Optional[str] # 验证模型给出的建议，可以传入reader模型进行二次检索
    sources: List[Dict[str, Any]] # 引用信息(QA大模型的回答引用来源)
    error: Optional[str] # 错误信息，如果发生错误，则需要处理错误
    iteration_count: int # 迭代次数，防止无限循环


class KnowledgeQAAgent:
    """知识库问答Agent"""

    def __init__(self, text_processor: Optional[TextProcessor] = None,
                 vector_store: Optional[VectorStore] = None):
        self.text_processor = text_processor or TextProcessor()
        self.vector_store = vector_store or VectorStore()   

        self.reader_llm = ReaderLLM() # 阅读本地文件资料的工具模型
        self.qa_llm = QALLM() # 知识库问答大模型
        self.finished_llm = FinishedLLM() # 判断智能体是否完成大模型回答
        self.verify_llm = VerifyLLM() # 验证资料是否足够的模型

        # 构建LangGraph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        log.info("KnowledgeQAAgent 初始化完成")

    def _build_graph(self) -> StateGraph:
        """构建reason-action架构的工作流图"""
        workflow = StateGraph(KnowledgeQAState)

        # 添加核心节点
        workflow.add_node("process_file", self._process_file_node)
        workflow.add_node("store_document", self._store_document_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("check_finished", self._check_finished_node)
        workflow.add_node("reader_search", self._reader_search_node)
        workflow.add_node("verify_context", self._verify_context_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # 设置边连接
        workflow.add_edge("process_file", "store_document")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "check_finished")
        workflow.add_edge("reader_search", "verify_context")
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

        # 完成检查后的条件路由
        workflow.add_conditional_edges(
            "check_finished",
            self._route_after_finished_check,
            {
                "finished": END,
                "need_search": "reader_search",
                "error": "handle_error"
            }
        )

        # 验证后的条件路由
        workflow.add_conditional_edges(
            "verify_context",
            self._route_after_verify,
            {
                "satisfied": "generate_answer",
                "need_more": "reader_search",
                "error": "handle_error"
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

    @traceable(name="check_finished_node")
    def _check_finished_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """检查回答是否完成的节点"""
        log.info("执行完成状态检查")
        
        try:
            query = state["query"]
            answer = state["qa_answer"]
            
            # 记录FinishedLLM的输入
            log.info("=" * 50)
            log.info("🔍 FinishedLLM 输入:")
            log.info(f"  问题: {query}")
            log.info(f"  当前回答: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            log.info("=" * 50)
            
            # 使用finished_llm判断是否完成
            finished_state = self.finished_llm.generate(query, answer)
            state["finished_state"] = finished_state
            
            # 处理返回的字典格式
            if isinstance(finished_state, dict):
                finished = finished_state.get("finished", False)
                reason = finished_state.get("reason", "")
            else:
                finished = finished_state.finished
                reason = finished_state.reason
            
            # 记录FinishedLLM的输出
            log.info("=" * 50)
            log.info("🔍 FinishedLLM 输出:")
            log.info(f"  完成状态: {finished}")
            log.info(f"  判断原因: {reason}")
            log.info("=" * 50)
                
            log.info(f"完成状态检查结果: {finished}, 原因: {reason}")
            
        except Exception as e:
            log.error(f"完成状态检查失败: {e}")
            state["error"] = f"完成状态检查失败: {str(e)}"
        
        return state

    @traceable(name="reader_search_node")
    def _reader_search_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """Reader搜索节点"""
        log.info("执行Reader搜索")
        
        try:
            query = state["query"]
            suggestions = state.get("suggestions")
            
            # 记录ReaderLLM的输入
            log.info("=" * 50)
            log.info("📚 ReaderLLM 输入:")
            if suggestions:
                log.info(f"  搜索建议: {suggestions}")
                log.info(f"  原始问题: {query}")
            else:
                log.info(f"  搜索问题: {query}")
            log.info("=" * 50)
            
            # 使用reader_llm搜索文档片段
            if suggestions:
                # 如果有建议，使用建议进行二次搜索
                reader_result = self.reader_llm.generate(suggestions=suggestions)
            else:
                # 首次搜索
                reader_result = self.reader_llm.generate(query=query)
            
            state["reader_result"] = reader_result
            
            # 记录ReaderLLM的输出
            log.info("=" * 50)
            log.info("📚 ReaderLLM 输出:")
            log.info(f"  找到片段数量: {len(reader_result.fragments)}")
            for i, fragment in enumerate(reader_result.fragments, 1):
                log.info(f"  片段{i}: {fragment.filename} (行{fragment.start_line}-{fragment.end_line})")
                log.info(f"    内容: {fragment.content[:100]}{'...' if len(fragment.content) > 100 else ''}")
            log.info("=" * 50)
            
            log.info(f"Reader搜索完成，找到 {len(reader_result.fragments)} 个片段")
            
        except Exception as e:
            log.error(f"Reader搜索失败: {e}")
            state["error"] = f"Reader搜索失败: {str(e)}"
        
        return state

    @traceable(name="verify_context_node")
    def _verify_context_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """验证上下文节点"""
        log.info("执行上下文验证")
        
        try:
            query = state["query"]
            answer = state["qa_answer"]
            reader_result = state["reader_result"]
            
            # 构建上下文信息
            context_text = "\n".join([fragment.content for fragment in reader_result.fragments])
            
            # 记录VerifyLLM的输入
            log.info("=" * 50)
            log.info("✅ VerifyLLM 输入:")
            log.info(f"  问题: {query}")
            log.info(f"  当前回答: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            log.info(f"  检索到的上下文: {context_text[:200]}{'...' if len(context_text) > 200 else ''}")
            log.info("=" * 50)
            
            # 使用verify_llm验证资料是否足够
            verify_state = self.verify_llm.generate(query, answer, context_text)
            state["verify_state"] = verify_state
            
            # 处理字典格式的返回值
            if isinstance(verify_state, dict):
                state["suggestions"] = verify_state.get("suggestions")
                satisfied = verify_state.get("satisfied", False)
                reason = verify_state.get("reason", "")
            else:
                state["suggestions"] = verify_state.suggestions
                satisfied = verify_state.satisfied
                reason = verify_state.reason
            
            # 记录VerifyLLM的输出
            log.info("=" * 50)
            log.info("✅ VerifyLLM 输出:")
            log.info(f"  是否满足: {satisfied}")
            log.info(f"  判断原因: {reason}")
            if state["suggestions"]:
                log.info(f"  改进建议: {state['suggestions']}")
            log.info("=" * 50)
            
            log.info(f"验证结果: {satisfied}, 原因: {reason}")
            
        except Exception as e:
            log.error(f"上下文验证失败: {e}")
            state["error"] = f"上下文验证失败: {str(e)}"
        
        return state

    def _route_after_finished_check(self, state: KnowledgeQAState) -> str:
        """完成检查后的路由决策"""
        if state.get("error"):
            return "error"
        
        finished_state = state.get("finished_state")
        if not finished_state:
            return "error"
        
        # 检查迭代次数，防止无限循环
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= 5:  # 最多5次迭代
            log.warning("达到最大迭代次数，强制结束")
            return "finished"
        
        # 处理字典格式的返回值
        if isinstance(finished_state, dict):
            finished = finished_state.get("finished", False)
        else:
            finished = finished_state.finished
            
        if finished:
            return "finished"
        else:
            # 需要进一步搜索
            return "need_search"

    def _route_after_verify(self, state: KnowledgeQAState) -> str:
        """验证后的路由决策"""
        if state.get("error"):
            return "error"
        
        verify_state = state.get("verify_state")
        if not verify_state:
            return "error"
        
        # 处理字典格式的返回值
        if isinstance(verify_state, dict):
            satisfied = verify_state.get("satisfied", False)
        else:
            satisfied = verify_state.satisfied
            
        if satisfied:
            # 资料足够，重新生成回答
            return "satisfied"
        else:
            # 需要更多资料
            return "need_more"

    @traceable(name="process_file_node")
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
            text = FileParser.parse_file(file_path)
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

    @traceable(name="store_document_node")
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
            self.vector_store.add_documents(documents, batch_size=10)
            log.info("向量化入库完成")

            # 保存向量库
            self.vector_store.save_vector_store()
            log.info("向量库保存完成")

        except Exception as e:
            log.error(f"文档存储失败: {e}")
            state["error"] = f"文档存储失败: {str(e)}"

        return state

    @traceable(name="retrieve_context_node")
    def _retrieve_context_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """上下文检索节点"""
        log.info("执行上下文检索")

        try:
            query = state["query"]
            if not query or not query.strip():
                state["error"] = "查询内容为空"
                return state

            context_docs = self.vector_store.similarity_search(query, k=settings.search_k)
            state["context_docs"] = context_docs

            log.info(f"检索到 {len(context_docs)} 个相关文档")

        except Exception as e:
            log.error(f"上下文检索失败: {e}")
            state["error"] = f"上下文检索失败: {str(e)}"
            state["context_docs"] = []

        return state

    @traceable(name="generate_answer_node")
    def _generate_answer_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """答案生成节点"""
        log.info("执行答案生成")

        try:
            query = state["query"]
            context_docs = state["context_docs"]
            reader_result = state.get("reader_result")

            # 记录QALLM的输入
            log.info("=" * 50)
            log.info("🤖 QALLM 输入:")
            log.info(f"  问题: {query}")
            log.info(f"  向量检索文档数量: {len(context_docs)}")
            if reader_result and reader_result.fragments:
                log.info(f"  Reader检索片段数量: {len(reader_result.fragments)}")
                for i, fragment in enumerate(reader_result.fragments, 1):
                    log.info(f"    片段{i}: {fragment.filename} (行{fragment.start_line}-{fragment.end_line})")
            else:
                log.info("  Reader检索片段: 无")
            log.info("=" * 50)

            # 调用LLM生成回答，传入reader_result
            result = self.qa_llm.generate(query, context_docs, reader_result)

            state["qa_answer"] = result["answer"]
            state["sources"] = result["sources"]

            # 记录QALLM的输出
            log.info("=" * 50)
            log.info("🤖 QALLM 输出:")
            log.info(f"  生成回答: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
            log.info(f"  引用来源数量: {len(result['sources'])}")
            for i, source in enumerate(result["sources"], 1):
                log.info(f"    来源{i}: {source.get('content', '')[:50]}{'...' if len(source.get('content', '')) > 50 else ''}")
            log.info("=" * 50)

            # 结果后处理
            answer = state["qa_answer"]
            if len(answer) < 10:
                state["qa_answer"] = "抱歉，我无法提供完整的回答。"

            # 确保引用格式正确
            sources = state["sources"]
            for i, source in enumerate(sources, 1):
                source["index"] = i

            # 增加迭代次数
            state["iteration_count"] = state.get("iteration_count", 0) + 1

            log.info("答案生成完成")

        except Exception as e:
            log.error(f"答案生成失败: {e}")
            state["error"] = f"答案生成失败: {str(e)}"
            state["qa_answer"] = "抱歉，我无法回答这个问题。"
            state["sources"] = []

        return state

    @traceable(name="handle_error_node")
    def _handle_error_node(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """错误处理节点"""
        log.error("执行错误处理")

        error_msg = state.get("error", "未知错误")
        log.error(f"处理错误: {error_msg}")

        # 设置默认错误响应
        state["answer"] = f"处理过程中发生错误: {error_msg}"
        state["sources"] = []

        return state

    @traceable(name="chat")
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
            reader_result=None,
            qa_answer="",
            verify_state=None,
            finished_state=None,
            suggestions=None,
            sources=[],
            error=None,
            iteration_count=0
        )

        # 执行图
        final_state = self.app.invoke(initial_state)

        # 返回结果
        result = {
            "answer": final_state["qa_answer"],
            "sources": final_state["sources"],
            "mode": final_state["mode"],
            "iteration_count": final_state.get("iteration_count", 0),
            "finished_state": final_state.get("finished_state"),
            "verify_state": final_state.get("verify_state")
        }

        log.info("输入处理完成")
        return result

    @traceable(name="chat_streaming")
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
            reader_result=None,
            qa_answer="",
            verify_state=None,
            finished_state=None,
            suggestions=None,
            sources=[],
            error=None,
            iteration_count=0
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
        for chunk in self.qa_llm.streaming(query, state["context_docs"]):
            if isinstance(chunk, dict):
                # 最后的元数据，添加 mode 信息
                chunk["mode"] = mode
                yield chunk
            else:
                # 流式文本内容
                yield chunk

    def clear_memory(self):
        """清空各LLM的记忆"""
        self.qa_llm.clear_memory()
        self.reader_llm.clear_memory()
        log.info("各LLM记忆已清空")


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.agent
    print("="*100)
    print("知识库问答Agent - 端到端测试")
    print("="*100)

    print("1. 初始化Agent")
    agent = KnowledgeQAAgent()
    print("   Agent初始化完成\n")

    # print("2. 测试文件上传处理")
    # file_path = "examples/凡人修仙传test.txt"
    # print(f"   上传文件: {file_path}")
    # result_upload = agent.chat("", file_path=file_path)
    # print(f"   模式: {result_upload['mode']}\n")

    print("3. 测试纯查询对话")
    query1 = "韩立是如何进入七玄门的？记名弟子初次考验包含哪些关键路段与环节？"
    query1 = "墨大夫的真实来历与核心目的是什么？他与韩立关系的转折点发生在哪些事件上？"
    query1 = "神手谷中的神秘小瓶具备什么规律与用途？韩立如何验证并应用？"
    query1 = "落日峰“死契血斗”前后的关键人物与转折是什么？韩立如何扭转战局？"
    query1 = "韩立已系统掌握的法术与其局限是什么？他如何“法武并用”？"
    # query1 = "人类（Human）的标准种族特性里，能力值（Ability Scores）如何改变？人类还会获得哪些语言？"
    # query1 = "在战斗中当你使用一次“动作（Action）”时，下面哪项不是标准动作？（A）Dash（冲刺） （B）Disengage（脱离） （C）Dodge（躲闪） （D）Teleport（瞬移）"
    # query1 = "如果你在某回合用 bonus action（奖励动作）施放了一个法术，你还能在同一回合再施放一个非戏法（cantrip）的法术吗？为什么？"
    # query1 = "简述短休息（Short Rest）与长休息（Long Rest）的主要区别与效果（至少包含各自持续时间与恢复内容）。"
    # query1 = "当一个目标处于 Grappled（被擒抱/缠住）状态时，会发生什么？列出该状态对目标的具体机械效果。"
    print(f"   Q: {query1}")
    result1 = agent.chat(query1)
    print(f"   A: {result1['answer']}")
    print(f"   模式: {result1['mode']}\n")

    print("\n✅ Agent测试完成!")
