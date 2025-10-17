"""文本分段和向量化"""

from typing import Optional, List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable

from .config import settings
from .log_manager import log


class TextProcessor:
    """文本分段和向量化处理器"""

    def __init__(self) -> None:
        # 创建分段器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # 创建嵌入模型
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.embedding_model
        )

        # 向量存储（延迟初始化）
        self.vector_store: Optional[FAISS] = None
        self._try_load_existing_vector_store()

    def _try_load_existing_vector_store(self) -> None:
        """尝试加载现有的向量存储"""
        try:
            persist_dir = Path(settings.vector_store_path)
            index_path = persist_dir / "index.faiss"
            if index_path.exists():
                self.vector_store = FAISS.load_local(
                    str(persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name="index")
                log.info(f"加载现有的向量存储: {persist_dir}")
            else:
                log.info("未找到现有向量存储，将在首次添加文档时创建")
        except Exception as e:
            log.warning(f"加载向量存储失败: {e}，将在首次添加文档时创建新的")

    def split_text(self, text: str) -> List[Document]:
        """分段文本"""
        return self.splitter.create_documents([text])

    @traceable(name="add_documents")
    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """添加文档到向量存储

        Args:
            documents: 要添加的文档列表
            batch_size: 首次创建向量存储时的批次大小，默认50
        """
        try:
            if not documents:
                log.warning("文档列表为空，跳过添加")
                return

            total_docs = len(documents)
            log.info(f"准备添加 {total_docs} 个文档")

            # 如果向量存储不存在，使用第一批创建
            if self.vector_store is None:
                first_batch = documents[:batch_size]
                log.info(f"首次添加文档，使用前 {len(first_batch)} 个文档创建向量存储")
                self.vector_store = FAISS.from_documents(
                    first_batch, self.embeddings)
                log.info(f"✅ 向量存储创建成功，已添加第 1 批 ({len(first_batch)} 个文档)")

                # 处理剩余文档 - 一次性全部添加
                remaining_docs = documents[batch_size:]
                if remaining_docs:
                    log.info(f"正在一次性添加剩余 {len(remaining_docs)} 个文档...")
                    self.vector_store.add_documents(remaining_docs)
                    log.info(f"✅ 剩余文档添加成功")
            else:
                # 向量存储已存在，一次性添加所有文档
                log.info(f"向量存储已存在，正在一次性添加 {total_docs} 个文档...")
                self.vector_store.add_documents(documents)
                log.info(f"✅ 所有文档添加成功")

            log.info(f"🎉 所有文档添加完成！总计 {total_docs} 个文档")

        except Exception as e:
            log.error(f"❌ 添加文档到向量存储失败: {e}")
            raise e

    def save_vector_store(self) -> None:
        """保存向量存储到磁盘"""
        if self.vector_store is None:
            log.warning("向量存储为空，无需保存")
            return

        try:
            persist_dir = Path(settings.vector_store_path)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(
                str(persist_dir), index_name="index")
            log.info(f"向量存储已保存到: {persist_dir}")
        except Exception as e:
            log.error(f"保存向量存储失败: {e}")
            raise e

    @traceable(name="similarity_search")
    def similarity_search(self, query: str, k: int = settings.search_k) -> List[Document]:
        """相似度搜索"""
        if self.vector_store is None:
            log.error("向量存储未初始化，请先添加文档")
            raise ValueError("向量存储未初始化")
        return self.vector_store.similarity_search(query, k, filter=None)


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.text_processor
    print("="*100)
    print("开始测试文本处理和向量化")
    print("="*100)
    
    from .file_parser import TextFileParser
    
    # 步骤1: 解析文件
    print("\n步骤1: 解析文件")
    text = TextFileParser.parse_file("examples/三国演义.txt")
    print(f"文件解析成功，文本长度: {len(text)} 字符")
    print(f"文本预览: {text[:100]}...\n")
    
    # 步骤2: 初始化处理器并分段
    print("步骤2: 初始化处理器并分段")
    text_processor = TextProcessor()
    documents: List[Document] = text_processor.split_text(text)
    print(f"文档已分段，共 {len(documents)} 段")
    print(f"\n前3段预览：")
    for i, doc in enumerate(documents[:3], 1):
        print(f"\n第 {i} 段:")
        print(doc.page_content[:200] + "...")
        print("-"*100)
    
    # 步骤3: 添加文档到向量存储（使用小批量测试）
    print("\n步骤3: 添加文档到向量存储")
    test_batch_size = 10  # 先测试10个文档
    print(f"注意：为了快速测试，仅使用前 {test_batch_size} 段文档")
    text_processor.add_documents(documents[:test_batch_size], batch_size=5)
    
    # 步骤4: 保存向量存储
    print("\n步骤4: 保存向量存储")
    text_processor.save_vector_store()
    
    # 步骤5: 测试相似度搜索
    print("\n步骤5: 测试相似度搜索")
    query = "临江仙"
    print(f"搜索关键词: {query}")
    results = text_processor.similarity_search(query, k=3)
    print(f"\n找到 {len(results)} 个相关结果：")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(result.page_content[:300] + "...")
        print("-"*100)
    
    print("\n" + "="*100)
    print("✅ 所有测试完成!")
    print("="*100)
