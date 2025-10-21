"""向量存储管理器 - FAISS向量数据库的定制化配置"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import shutil
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable

from .config import settings
from .log_manager import log


class VectorStore:
    """FAISS向量存储管理器 - 支持定制化配置"""
    
    def __init__(
        self,
        embeddings: Optional[OpenAIEmbeddings] = None,
        vector_store_path: Optional[str] = None,
        index_name: str = "index",
        allow_dangerous_deserialization: bool = True
    ):
        """初始化向量存储管理器
        
        Args:
            embeddings: 嵌入模型，默认使用配置中的模型
            vector_store_path: 向量存储路径，默认使用配置中的路径
            index_name: 索引名称
            allow_dangerous_deserialization: 是否允许危险的反序列化
        """
        self.index_name = index_name
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        
        # 设置路径
        self.vector_store_path = vector_store_path or settings.vector_store_path
        self.persist_dir = Path(self.vector_store_path)
        
        # 设置嵌入模型
        if embeddings is None:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.siliconcloud_api_key,
                openai_api_base=settings.siliconcloud_api_base,
                model=settings.embedding_model
            )
        else:
            self.embeddings = embeddings
            
        # 向量存储（延迟初始化）
        self._vector_store: Optional[FAISS] = None
        self._load_existing_vector_store()
    
    @property
    def vector_store(self) -> Optional[FAISS]:
        """获取向量存储实例"""
        return self._vector_store
    
    def _load_existing_vector_store(self) -> None:
        """加载现有的向量存储"""
        try:
            index_path = self.persist_dir / f"{self.index_name}.faiss"
            if index_path.exists():
                self._vector_store = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=self.allow_dangerous_deserialization,
                    index_name=self.index_name
                )
                log.info(f"✅ 向量存储加载成功: {self.persist_dir}")
            else:
                log.info(f"ℹ️ 未找到现有向量存储: {index_path}")
        except Exception as e:
            log.warning(f"⚠️ 加载向量存储失败: {e}")
    
    @traceable(name="create_vector_store")
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """创建新的向量存储"""
        if not documents:
            raise ValueError("文档列表不能为空")
        
        log.info(f"🚀 创建新的向量存储，文档数量: {len(documents)}")
        self._vector_store = FAISS.from_documents(documents, self.embeddings)
        log.info("✅ 向量存储创建成功")
        return self._vector_store
    
    @traceable(name="add_documents")
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 10,
        create_if_not_exists: bool = True
    ) -> None:
        """添加文档到向量存储"""
        if not documents:
            log.warning("⚠️ 文档列表为空，跳过添加")
            return
        
        total_docs = len(documents)
        log.info(f"📝 准备添加 {total_docs} 个文档，批次大小: {batch_size}")
        
        # 如果向量存储不存在且允许创建
        if self._vector_store is None:
            if create_if_not_exists:
                first_batch = documents[:batch_size]
                log.info(f"🆕 创建新向量存储，使用前 {len(first_batch)} 个文档")
                self._vector_store = FAISS.from_documents(first_batch, self.embeddings)
                log.info(f"✅ 向量存储创建成功，已添加第 1 批 ({len(first_batch)} 个文档)")
                
                # 分批处理剩余文档
                remaining_docs = documents[batch_size:]
                if remaining_docs:
                    self._add_documents_in_batches(remaining_docs, batch_size, start_batch=2)
            else:
                raise ValueError("向量存储不存在且不允许创建新存储")
        else:
            # 向量存储已存在，分批添加所有文档
            log.info(f"📚 向量存储已存在，开始分批添加 {total_docs} 个文档")
            self._add_documents_in_batches(documents, batch_size, start_batch=1)
        
        log.info(f"🎉 所有文档添加完成！总计 {total_docs} 个文档")
    
    def _add_documents_in_batches(
        self, 
        documents: List[Document], 
        batch_size: int, 
        start_batch: int = 1
    ) -> None:
        """分批添加文档的内部方法"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + start_batch
            log.info(f"⏳ 正在处理第 {batch_num} 批 ({len(batch)} 个文档)")
            self._vector_store.add_documents(batch)
            log.info(f"✅ 第 {batch_num} 批添加成功")
    
    @traceable(name="similarity_search")
    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20
    ) -> List[Document]:
        """相似度搜索"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化，请先添加文档")
        
        k = k or settings.search_k
        return self._vector_store.similarity_search(
            query, k=k, filter=filter, fetch_k=fetch_k
        )
    
    @traceable(name="similarity_search_with_score")
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """带分数的相似度搜索"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化，请先添加文档")
        
        k = k or settings.search_k
        return self._vector_store.similarity_search_with_score(
            query, k=k, filter=filter
        )
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """保存向量存储到磁盘
        
        Args:
            path: 保存路径，默认使用初始化时的路径
        """
        if self._vector_store is None:
            log.warning("⚠️ 向量存储为空，无需保存")
            return
        
        save_path = path or self.vector_store_path
        persist_dir = Path(save_path)
        
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(persist_dir), index_name=self.index_name)
            log.info(f"💾 向量存储已保存到: {persist_dir}")
        except Exception as e:
            log.error(f"❌ 保存向量存储失败: {e}")
            raise e
    
    def load_vector_store(self, path: Optional[str] = None) -> None:
        """从磁盘加载向量存储
        
        Args:
            path: 加载路径，默认使用初始化时的路径
        """
        load_path = path or self.vector_store_path
        persist_dir = Path(load_path)
        
        try:
            index_path = persist_dir / f"{self.index_name}.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"向量存储文件不存在: {index_path}")
            
            self._vector_store = FAISS.load_local(
                str(persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=self.allow_dangerous_deserialization,
                index_name=self.index_name
            )
            log.info(f"📂 向量存储加载成功: {persist_dir}")
        except Exception as e:
            log.error(f"❌ 加载向量存储失败: {e}")
            raise e
    
    def clear_vector_store(self) -> bool:
        """清除向量数据库"""
        log.info("🚀 开始执行向量数据库清除操作")
        
        try:
            # 清除内存中的向量存储
            vector_store_status = "存在" if self._vector_store is not None else "不存在"
            log.info(f"📊 清除前内存向量存储状态: {vector_store_status}")
            
            self._vector_store = None
            log.info("✅ 内存中的向量存储已清除")
            
            # 检查磁盘上的向量数据库文件
            log.info(f"📁 向量数据库路径: {self.persist_dir}")
            log.info(f"📊 目录是否存在: {'是' if self.persist_dir.exists() else '否'}")
            
            if self.persist_dir.exists():
                try:
                    files_in_dir = list(self.persist_dir.iterdir())
                    log.info(f"📋 目录中的文件: {[f.name for f in files_in_dir]}")
                except Exception as e:
                    log.warning(f"⚠️ 无法列出目录内容: {e}")
                
                shutil.rmtree(self.persist_dir)
                log.info(f"✅ 磁盘向量数据库已删除: {self.persist_dir}")
            else:
                log.info("ℹ️ 向量数据库目录不存在，无需删除")
            
            # 重新创建空的目录
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"✅ 已重新创建向量数据库目录: {self.persist_dir}")
            
            # 验证目录创建成功
            if self.persist_dir.exists():
                log.info("✅ 验证: 向量数据库目录创建成功")
            else:
                log.error("❌ 验证: 向量数据库目录创建失败")
                return False
            
            log.info("🎉 向量数据库清除完成！")
            return True
            
        except Exception as e:
            log.error(f"❌ 清除向量数据库失败: {e}")
            log.error(f"❌ 错误类型: {type(e).__name__}")
            import traceback
            log.error(f"❌ 错误堆栈: {traceback.format_exc()}")
            return False
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """获取向量存储信息"""
        info = {
            "vector_store_exists": self._vector_store is not None,
            "persist_dir": str(self.persist_dir),
            "persist_dir_exists": self.persist_dir.exists(),
            "index_name": self.index_name,
            "embedding_model": getattr(self.embeddings, 'model', 'unknown'),
        }
        
        if self.persist_dir.exists():
            try:
                files = list(self.persist_dir.iterdir())
                info["files_in_directory"] = [f.name for f in files]
                info["index_file_exists"] = (self.persist_dir / f"{self.index_name}.faiss").exists()
                info["pkl_file_exists"] = (self.persist_dir / f"{self.index_name}.pkl").exists()
            except Exception as e:
                info["directory_access_error"] = str(e)
        
        return info
    
    def is_ready(self) -> bool:
        """检查向量存储是否就绪"""
        return self._vector_store is not None


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.vector_store
    """测试向量存储管理器"""
    print("="*100)
    print("开始测试向量存储管理器")
    print("="*100)
    
    from .file_parser import TextFileParser
    
    # 创建向量存储管理器
    vector_store = VectorStore()
    
    # 显示初始信息
    info = vector_store.get_vector_store_info()
    print(f"\n📊 向量存储信息:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 测试文档加载和分段
    print(f"\n📖 加载测试文档...")
    text = TextFileParser.parse_file("examples/三国演义.txt")
    print(f"文档长度: {len(text)} 字符")
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    documents = splitter.create_documents([text])
    print(f"文档分段数量: {len(documents)}")
    
    # 测试添加文档
    print(f"\n📝 测试添加文档...")
    test_docs = documents[:5]  # 只测试前5个文档
    vector_store.add_documents(test_docs, batch_size=3)
    
    # 测试保存
    print(f"\n💾 测试保存向量存储...")
    vector_store.save_vector_store()
    
    # 测试搜索
    print(f"\n🔍 测试相似度搜索...")
    query = "临江仙"
    results = vector_store.similarity_search(query, k=2)
    print(f"搜索关键词: {query}")
    print(f"找到 {len(results)} 个相关结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(result.page_content[:200] + "...")
    
    # 测试带分数搜索
    print(f"\n📊 测试带分数搜索...")
    results_with_score = vector_store.similarity_search_with_score(query, k=2)
    for i, (doc, score) in enumerate(results_with_score, 1):
        print(f"结果 {i} (分数: {score:.4f}): {doc.page_content[:100]}...")
    
    print("\n" + "="*100)
    print("✅ 向量存储管理器测试完成!")
    print("="*100)
