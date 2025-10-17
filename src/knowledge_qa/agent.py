"""LangGraph Agent 集成"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .text_processor import TextProcessor
from .llm import LLM
from .memory import MemoryManager
from .log_manager import log


class KnowledgeQAState(TypedDict):
    """知识库问答状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    context_docs: List[Document]
    answer: str
    sources: List[Dict[str, Any]]
    use_knowledge_base: bool
    retrieval_score: float


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
        """构建LangGraph执行流程图"""
        workflow = StateGraph(KnowledgeQAState)
        
        # 添加节点
        workflow.add_node("query_understanding", self._query_understanding)
        workflow.add_node("knowledge_retrieval", self._knowledge_retrieval)
        workflow.add_node("context_building", self._context_building)
        workflow.add_node("llm_generation", self._llm_generation)
        workflow.add_node("result_postprocessing", self._result_postprocessing)
        workflow.add_node("memory_update", self._memory_update)
        
        # 设置入口点
        workflow.set_entry_point("query_understanding")
        
        # 添加边
        workflow.add_edge("query_understanding", "knowledge_retrieval")
        workflow.add_edge("knowledge_retrieval", "context_building")
        workflow.add_edge("context_building", "llm_generation")
        workflow.add_edge("llm_generation", "result_postprocessing")
        workflow.add_edge("result_postprocessing", "memory_update")
        workflow.add_edge("memory_update", END)
        
        return workflow
    
    def _query_understanding(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """查询理解节点"""
        log.info("执行查询理解")
        
        query = state["query"]
        
        # 简单的查询分析逻辑
        knowledge_keywords = ["什么", "如何", "为什么", "怎么", "谁", "哪里", "什么时候"]
        use_knowledge_base = any(keyword in query for keyword in knowledge_keywords)
        
        # 如果查询太短或太简单，可能不需要知识库
        if len(query) < 3:
            use_knowledge_base = False
        
        state["use_knowledge_base"] = use_knowledge_base
        log.info(f"查询分析结果 - 使用知识库: {use_knowledge_base}")
        
        return state
    
    def _knowledge_retrieval(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """知识检索节点"""
        log.info("执行知识检索")
        
        if not state["use_knowledge_base"]:
            state["context_docs"] = []
            state["retrieval_score"] = 0.0
            return state
        
        try:
            query = state["query"]
            context_docs = self.text_processor.similarity_search(query, k=3)
            
            # 计算检索质量分数（简单实现）
            retrieval_score = min(len(context_docs) / 3.0, 1.0)
            
            state["context_docs"] = context_docs
            state["retrieval_score"] = retrieval_score
            
            log.info(f"检索到 {len(context_docs)} 个相关文档，质量分数: {retrieval_score:.2f}")
            
        except Exception as e:
            log.error(f"知识检索失败: {e}")
            state["context_docs"] = []
            state["retrieval_score"] = 0.0
        
        return state
    
    def _context_building(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """上下文构建节点"""
        log.info("执行上下文构建")
        
        # 这里主要是准备给LLM的上下文
        # 实际的上下文构建在LLM.generate()中完成
        return state
    
    def _llm_generation(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """LLM生成节点"""
        log.info("执行LLM生成")
        
        try:
            query = state["query"]
            context_docs = state["context_docs"]
            
            # 调用LLM生成回答
            result = self.llm.generate(query, context_docs, use_memory=True)
            
            state["answer"] = result["answer"]
            state["sources"] = result["sources"]
            
            log.info("LLM生成完成")
            
        except Exception as e:
            log.error(f"LLM生成失败: {e}")
            state["answer"] = "抱歉，我无法回答这个问题。"
            state["sources"] = []
        
        return state
    
    def _result_postprocessing(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """结果后处理节点"""
        log.info("执行结果后处理")
        
        # 添加质量检查
        answer = state["answer"]
        if len(answer) < 10:
            state["answer"] = "抱歉，我无法提供完整的回答。"
        
        # 确保引用格式正确
        sources = state["sources"]
        for i, source in enumerate(sources, 1):
            source["index"] = i
        
        log.info("结果后处理完成")
        return state
    
    def _memory_update(self, state: KnowledgeQAState) -> KnowledgeQAState:
        """记忆更新节点"""
        log.info("执行记忆更新")
        
        # 记忆更新已在LLM.generate()中完成
        # 这里可以添加额外的记忆管理逻辑
        
        return state
    
    def chat(self, query: str) -> Dict[str, Any]:
        """对话接口"""
        log.info(f"开始处理用户查询: {query}")
        
        # 初始化状态
        initial_state = KnowledgeQAState(
            messages=[HumanMessage(content=query)],
            query=query,
            context_docs=[],
            answer="",
            sources=[],
            use_knowledge_base=True,
            retrieval_score=0.0
        )
        
        # 执行图
        final_state = self.app.invoke(initial_state)
        
        # 返回结果
        result = {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "use_knowledge_base": final_state["use_knowledge_base"],
            "retrieval_score": final_state["retrieval_score"]
        }
        
        log.info("查询处理完成")
        return result
    
    def chat_streaming(self, query: str):
        """流式对话接口"""
        log.info(f"开始流式处理用户查询: {query}")
        
        # 对于流式输出，我们直接使用LLM的流式接口
        # 但先进行查询理解和知识检索
        state = KnowledgeQAState(
            messages=[HumanMessage(content=query)],
            query=query,
            context_docs=[],
            answer="",
            sources=[],
            use_knowledge_base=True,
            retrieval_score=0.0
        )
        
        # 执行前几个节点
        state = self._query_understanding(state)
        state = self._knowledge_retrieval(state)
        
        # 使用LLM的流式接口
        for chunk in self.llm.generate_streaming(query, state["context_docs"], use_memory=True):
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
    
    from .file_parser import TextFileParser
    
    print("知识库问答Agent - 端到端测试\n")
    
    print("1. 初始化Agent")
    agent = KnowledgeQAAgent()
    
    print("2. 加载知识库")
    text = TextFileParser.parse_file("examples/三国演义.txt")
    documents = agent.text_processor.split_text(text)
    agent.text_processor.add_documents(documents[:20], batch_size=10)
    print(f"   知识库加载完成，共 {len(documents[:20])} 段\n")
    
    print("3. 测试普通对话")
    query1 = "三国演义开头的那首词叫什么名字？"
    print(f"   Q: {query1}")
    result1 = agent.chat(query1)
    print(f"   A: {result1['answer']}")
    print(f"   使用知识库: {result1['use_knowledge_base']}, 检索分数: {result1['retrieval_score']:.2f}\n")
    
    print("4. 测试上下文理解")
    query2 = "它的作者是谁？"
    print(f"   Q: {query2}")
    result2 = agent.chat(query2)
    print(f"   A: {result2['answer']}\n")
    
    print("5. 测试流式输出")
    query3 = "刘备有什么特点？"
    print(f"   Q: {query3}")
    print("   A: ", end="", flush=True)
    for chunk in agent.chat_streaming(query3):
        if not isinstance(chunk, dict):
            print(chunk, end="", flush=True)
    print("\n")
    
    print("6. 测试记忆管理")
    print(f"   清空前消息数: {len(agent.llm.memory.get_messages())}")
    agent.clear_memory()
    print(f"   清空后消息数: {len(agent.llm.memory.get_messages())}")
    
    print("\n✅ Agent测试完成!")