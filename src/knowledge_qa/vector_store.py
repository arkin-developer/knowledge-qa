"""向量存储管理器"""


import faiss
import shutil
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS as LangChainFAISS
from langchain_community.docstore import InMemoryDocstore
from langsmith import traceable

from .config import settings
from .log_manager import log


class VectorStore:
    """FAISS向量存储管理器"""

    def __init__(
        self,
        embeddings: Optional[OpenAIEmbeddings] = None,
        vector_store_path: Optional[str] = None,
        index_name: str = "index"
    ):
        # 初始化配置
        self.index_name = index_name
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

        self._vector_store: Optional[FAISS] = None
        self._load_existing_vector_store()

    @property
    def vector_store(self) -> Optional[FAISS]:
        return self._vector_store

    def _load_existing_vector_store(self) -> None:
        """加载现有向量存储"""
        try:
            index_path = self.persist_dir / f"{self.index_name}.faiss"
            if index_path.exists():
                self._vector_store = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name=self.index_name
                )
                log.info(f"向量存储加载成功: {self.persist_dir}")
            else:
                log.info(f"未找到现有向量存储: {index_path}")
        except Exception as e:
            log.warning(f"加载向量存储失败: {e}")

    def _check_and_split_long_documents(self, documents: List[Document]) -> List[Document]:
        """检查文档长度并进行二次切分"""
        # from .config import settings

        # processed_documents = []
        # chunk_size = settings.chunk_size

        # for doc in documents:
        #     content_length = len(doc.page_content)

        #     if content_length <= chunk_size:
        #         processed_documents.append(doc)
        #     else:
        #         log.info(f"发现超长文档，长度: {content_length}，超过阈值: {chunk_size}，进行二次切分")

        #         from .text_processor import TextProcessor
        #         text_processor = TextProcessor(chunk_size=chunk_size)
        #         split_docs = text_processor.split_text(doc.page_content)

        #         for split_doc in split_docs:
        #             split_doc.metadata.update(doc.metadata)
        #             split_doc.metadata["original_length"] = content_length
        #             split_doc.metadata["split_from"] = "long_document"

        #         processed_documents.extend(split_docs)
        #         log.info(f"超长文档切分为 {len(split_docs)} 个片段")

        return documents

    @traceable(name="create_vector_store")
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """创建新的向量存储，使用IndexFlatIP索引（适合标准化向量）"""
        if not documents:
            raise ValueError("文档列表不能为空")

        log.info(f"创建新的向量存储，文档数量: {len(documents)}")

        try:
            import faiss

            # 获取文档文本
            texts = [doc.page_content for doc in documents]

            # 获取嵌入向量
            log.info("开始生成嵌入向量...")
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings_list).astype('float32')

            # 创建IndexFlatIP索引（内积索引，适合标准化向量）
            dimension = len(embeddings_array[0])
            index = faiss.IndexFlatIP(dimension)
            log.info(f"创建IndexFlatIP索引，维度: {dimension}")

            # 添加向量到索引
            index.add(embeddings_array)
            log.info(f"向量已添加到索引，总数: {index.ntotal}")

            # 创建docstore和index_to_docstore_id映射
            docstore_dict = {}
            index_to_docstore_id = {}

            for i, doc in enumerate(documents):
                doc_id = str(i)  # 使用简单的数字ID
                docstore_dict[doc_id] = doc
                index_to_docstore_id[i] = doc_id

            # 创建Docstore对象
            docstore = InMemoryDocstore(docstore_dict)

            # 创建LangChain FAISS包装器
            self._vector_store = LangChainFAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )

            log.info("向量存储创建成功 (使用IndexFlatIP内积索引)")
            return self._vector_store

        except Exception as e:
            log.error(f"创建向量存储失败: {e}")
            # 如果失败，回退到默认方法
            log.info("回退到默认L2距离索引")
            self._vector_store = FAISS.from_documents(
                documents, self.embeddings)
            log.info("向量存储创建成功 (使用默认L2距离索引)")
            return self._vector_store

    @traceable(name="add_documents")
    def add_documents(self, documents: List[Document], source_info: Dict[str, Any] = None, batch_size: int = 50) -> None:
        """添加文档到向量存储（分批处理）"""
        if not documents:
            log.warning("文档列表为空，跳过添加")
            return

        # 检查文档长度并进行二次切分
        processed_documents = self._check_and_split_long_documents(documents)

        total_docs = len(processed_documents)
        log.info(f"开始分批添加文档，总计 {total_docs} 个文档，每批 {batch_size} 个")

        # 分批处理文档
        for i in range(0, total_docs, batch_size):
            batch_documents = processed_documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            log.info(
                f"处理第 {batch_num}/{total_batches} 批，包含 {len(batch_documents)} 个文档")

            if self._vector_store is None:
                # 第一批文档，创建新的向量存储（使用IndexFlatIP）
                self.create_vector_store(batch_documents)
                self.save_vector_store()
            else:
                # 后续批次，添加到现有向量存储
                # 需要确保新添加的向量也使用内积索引
                try:

                    # 获取新文档的嵌入向量
                    texts = [doc.page_content for doc in batch_documents]
                    embeddings_list = self.embeddings.embed_documents(texts)
                    embeddings_array = np.array(
                        embeddings_list).astype('float32')

                    # 添加到现有索引
                    self._vector_store.index.add(embeddings_array)

                    # 更新docstore
                    start_idx = self._vector_store.index.ntotal - \
                        len(batch_documents)
                    for i, doc in enumerate(batch_documents):
                        doc_id = str(start_idx + i)
                        self._vector_store.docstore._dict[doc_id] = doc
                        self._vector_store.index_to_docstore_id[start_idx + i] = doc_id

                    log.info(f"第 {batch_num} 批文档已添加到IndexFlatIP索引")

                except Exception as e:
                    log.warning(f"使用IndexFlatIP添加文档失败，回退到默认方法: {e}")
                    self._vector_store.add_documents(batch_documents)

                self.save_vector_store()

            log.info(f"第 {batch_num} 批文档添加完成")

        log.info(f"所有文档添加完成，总计 {total_docs} 个文档")

    @traceable(name="similarity_search")
    def similarity_search(self, query: str, k: int = None, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """相似度搜索"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化，请先添加文档")

        k = k or settings.search_k
        results = self._vector_store.similarity_search(
            query, k=k, filter=filter)
        log.info(f"搜索完成: 查询='{query[:50]}...', 返回={len(results)} 个结果")
        return results

    @traceable(name="similarity_search_with_score")
    def similarity_search_with_score(self, query: str, k: int = None, filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """带分数的相似度搜索"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化，请先添加文档")

        k = k or settings.search_k
        results = self._vector_store.similarity_search_with_score(
            query, k=k, filter=filter)
        log.info(f"带分数搜索完成: 查询='{query[:50]}...', 返回={len(results)} 个结果")
        return results

    def save_vector_store(self, path: Optional[str] = None) -> None:
        """保存向量存储到磁盘"""
        if self._vector_store is None:
            log.warning("向量存储为空，无需保存")
            return

        save_path = path or self.vector_store_path
        persist_dir = Path(save_path)

        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(
                str(persist_dir), index_name=self.index_name)
            log.info(f"向量存储已保存到: {persist_dir}")
        except Exception as e:
            log.error(f"保存向量存储失败: {e}")
            raise e

    def load_vector_store(self, path: Optional[str] = None) -> None:
        """从磁盘加载向量存储"""
        load_path = path or self.vector_store_path
        persist_dir = Path(load_path)

        try:
            index_path = persist_dir / f"{self.index_name}.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"向量存储文件不存在: {index_path}")

            self._vector_store = FAISS.load_local(
                str(persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name=self.index_name
            )
            log.info(f"向量存储加载成功: {persist_dir}")
        except Exception as e:
            log.error(f"加载向量存储失败: {e}")
            raise e

    def clear_vector_store(self) -> bool:
        """清除向量数据库"""
        log.info("开始执行向量数据库清除操作")

        try:
            self._vector_store = None

            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
                log.info(f"磁盘向量数据库已删除: {self.persist_dir}")
            else:
                log.info("向量数据库目录不存在，无需删除")

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"已重新创建向量数据库目录: {self.persist_dir}")

            return True

        except Exception as e:
            log.error(f"清除向量数据库失败: {e}")
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

        if self._vector_store is not None:
            try:
                # 获取文档数量，兼容不同的docstore类型
                if hasattr(self._vector_store.docstore, '__len__'):
                    info["document_count"] = len(self._vector_store.docstore)
                elif hasattr(self._vector_store.docstore, '_dict'):
                    info["document_count"] = len(
                        self._vector_store.docstore._dict)
                else:
                    info["document_count"] = "未知"

                info["persist_path"] = str(self.persist_dir)
                info["index_ntotal"] = self._vector_store.index.ntotal
                info["index_dimension"] = self._vector_store.index.d
            except Exception as e:
                info["vector_store_error"] = str(e)

        if self.persist_dir.exists():
            try:
                files = list(self.persist_dir.iterdir())
                info["files_in_directory"] = [f.name for f in files]
                info["index_file_exists"] = (
                    self.persist_dir / f"{self.index_name}.faiss").exists()
                info["pkl_file_exists"] = (
                    self.persist_dir / f"{self.index_name}.pkl").exists()
            except Exception as e:
                info["directory_access_error"] = str(e)

        return info

    def is_ready(self) -> bool:
        """检查向量存储是否就绪"""
        return self._vector_store is not None

    def list_all_documents(self, limit: int = None) -> List[Document]:
        """列出所有文档记录"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化，请先添加文档")

        try:
            # 获取所有文档ID
            all_ids = list(self._vector_store.docstore._dict.keys())
            total_count = len(all_ids)

            log.info(f"向量存储中共有 {total_count} 个文档")

            # 如果指定了限制，只返回前N个
            if limit is not None:
                all_ids = all_ids[:limit]
                log.info(f"限制返回前 {limit} 个文档")

            # 获取文档内容
            documents = []
            for i, doc_id in enumerate(all_ids):
                try:
                    doc = self._vector_store.docstore._dict[doc_id]
                    documents.append(doc)
                    if (i + 1) % 100 == 0:
                        log.info(f"已处理 {i + 1}/{len(all_ids)} 个文档")
                except Exception as e:
                    log.warning(f"获取文档 {doc_id} 失败: {e}")

            log.info(f"成功获取 {len(documents)} 个文档")
            return documents

        except Exception as e:
            log.error(f"列出文档失败: {e}")
            raise e

    def print_all_documents(self, limit: int = None, show_content: bool = True, content_length: int = 200) -> None:
        """打印所有文档记录"""
        try:
            documents = self.list_all_documents(limit)

            print("=" * 100)
            print(f"向量存储文档列表 (共 {len(documents)} 个文档)")
            print("=" * 100)

            for i, doc in enumerate(documents, 1):
                print(f"\n[文档 {i}]")
                print(f"ID: {doc.metadata.get('id', 'unknown')}")
                print(f"来源: {doc.metadata.get('source', 'unknown')}")
                print(f"长度: {len(doc.page_content)} 字符")

                # 打印元数据
                if doc.metadata:
                    print("元数据:")
                    for key, value in doc.metadata.items():
                        if key != 'id':  # 避免重复显示ID
                            print(f"  {key}: {value}")

                # 打印内容（可选）
                if show_content:
                    content = doc.page_content
                    if len(content) > content_length:
                        content = content[:content_length] + "..."
                    print(f"内容预览: {content}")

                print("-" * 80)

        except Exception as e:
            log.error(f"打印文档失败: {e}")
            print(f"错误: {e}")

    def get_document_vector(self, doc_id: str = None, doc_index: int = 0) -> Optional[np.ndarray]:
        """获取指定文档的向量"""
        if self._vector_store is None:
            raise ValueError("向量存储未初始化")

        try:
            if doc_id is not None:
                # 通过ID获取文档
                if doc_id in self._vector_store.docstore._dict:
                    doc = self._vector_store.docstore._dict[doc_id]
                else:
                    raise ValueError(f"文档ID {doc_id} 不存在")
            else:
                # 通过索引获取文档
                all_ids = list(self._vector_store.docstore._dict.keys())
                if doc_index >= len(all_ids):
                    raise ValueError(
                        f"文档索引 {doc_index} 超出范围，总共有 {len(all_ids)} 个文档")
                doc_id = all_ids[doc_index]
                doc = self._vector_store.docstore._dict[doc_id]

            # 获取文档在FAISS索引中的位置
            # FAISS索引中的位置是通过文档在docstore中的顺序确定的
            all_ids = list(self._vector_store.docstore._dict.keys())
            faiss_index = all_ids.index(doc_id)

            # 获取文档的向量
            vector = self._vector_store.index.reconstruct(faiss_index)
            return vector

        except Exception as e:
            log.error(f"获取文档向量失败: {e}")
            raise e

    def check_vector_normalization(self, doc_id: str = None, doc_index: int = 0) -> Dict[str, Any]:
        """检查向量的标准化状态"""
        try:
            vector = self.get_document_vector(doc_id, doc_index)

            # 计算向量的L2范数（欧几里得范数）
            l2_norm = np.linalg.norm(vector)

            # 计算向量的L1范数（曼哈顿范数）
            l1_norm = np.linalg.norm(vector, ord=1)

            # 计算向量的最大值和最小值
            max_val = np.max(vector)
            min_val = np.min(vector)

            # 计算向量的均值和标准差
            mean_val = np.mean(vector)
            std_val = np.std(vector)

            # 检查是否标准化（L2范数应该接近1）
            is_normalized = abs(l2_norm - 1.0) < 1e-6

            result = {
                "vector_shape": vector.shape,
                "l2_norm": l2_norm,
                "l1_norm": l1_norm,
                "max_value": max_val,
                "min_value": min_val,
                "mean_value": mean_val,
                "std_value": std_val,
                "is_normalized": is_normalized,
                "normalization_tolerance": 1e-6,
                "vector_sample": vector[:10].tolist()  # 显示前10个值
            }

            return result

        except Exception as e:
            log.error(f"检查向量标准化失败: {e}")
            raise e

    def print_vector_analysis(self, doc_id: str = None, doc_index: int = 0) -> None:
        """打印向量分析结果"""
        try:
            analysis = self.check_vector_normalization(doc_id, doc_index)

            print("=" * 80)
            print("向量标准化分析")
            print("=" * 80)
            print(f"向量维度: {analysis['vector_shape']}")
            print(f"L2范数: {analysis['l2_norm']:.6f}")
            print(f"L1范数: {analysis['l1_norm']:.6f}")
            print(f"最大值: {analysis['max_value']:.6f}")
            print(f"最小值: {analysis['min_value']:.6f}")
            print(f"均值: {analysis['mean_value']:.6f}")
            print(f"标准差: {analysis['std_value']:.6f}")
            print(f"是否标准化: {'是' if analysis['is_normalized'] else '否'}")
            print(f"标准化容差: {analysis['normalization_tolerance']}")
            print(f"向量样本 (前10个值): {analysis['vector_sample']}")

            # 判断标准化状态
            if analysis['is_normalized']:
                print("\n✅ 向量已经标准化 (L2范数 ≈ 1)")
            else:
                print(f"\n❌ 向量未标准化 (L2范数 = {analysis['l2_norm']:.6f})")
                if analysis['l2_norm'] > 1.1:
                    print("   向量范数过大，可能需要标准化")
                elif analysis['l2_norm'] < 0.9:
                    print("   向量范数过小，可能需要标准化")

            print("=" * 80)

        except Exception as e:
            log.error(f"打印向量分析失败: {e}")
            print(f"错误: {e}")

    def get_index_type(self) -> str:
        """获取当前索引类型"""
        if self._vector_store is None:
            return "未初始化"

        try:
            index_type = type(self._vector_store.index).__name__
            return index_type
        except Exception as e:
            log.error(f"获取索引类型失败: {e}")
            return f"错误: {e}"

    def print_index_info(self) -> None:
        """打印索引信息"""
        if self._vector_store is None:
            print("向量存储未初始化")
            return

        try:
            index = self._vector_store.index
            index_type = type(index).__name__

            print("=" * 80)
            print("FAISS索引信息")
            print("=" * 80)
            print(f"索引类型: {index_type}")
            print(f"向量维度: {index.d}")
            print(f"向量总数: {index.ntotal}")
            print(f"是否已训练: {'是' if index.is_trained else '否'}")

            if "IndexFlatIP" in index_type:
                print("\n✅ 使用IndexFlatIP (内积索引)")
                print("   这是标准化向量的最佳选择！")
            elif "IndexFlatL2" in index_type:
                print("\n⚠️ 使用IndexFlatL2 (L2距离索引)")
                print("   对于标准化向量，内积索引更合适")
            else:
                print(f"\n❓ 索引类型: {index_type}")

            print("=" * 80)

        except Exception as e:
            log.error(f"打印索引信息失败: {e}")
            print(f"错误: {e}")


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.vector_store
    # print("="*100)
    # print("开始测试向量存储管理器")
    # print("="*100)
    # from .text_processor import TextProcessor
    # from .file_parser import FileParser

    # text = FileParser.parse_file("examples/凡人修仙传test.txt")
    # text_processor = TextProcessor()
    # documents = text_processor.split_text(text)
    # vector_store = VectorStore()
    # vector_store.add_documents(documents, batch_size=50)
    # print(vector_store.get_vector_store_info())

    # 创建向量存储实例
    # vector_store = VectorStore()

    # # 检查向量存储状态
    # info = vector_store.get_vector_store_info()
    # print("向量存储信息:")
    # for key, value in info.items():
    #     print(f"  {key}: {value}")

    # print("\n" + "="*100)

    # # 如果向量存储存在，列出所有文档
    # if vector_store.is_ready():
    #     print("开始列出所有文档记录...")

    #     # 只显示前10个文档的详细信息，避免输出过多
    #     vector_store.print_all_documents(limit=10, show_content=True, content_length=150)

    #     # 如果需要查看所有文档的统计信息
    #     all_docs = vector_store.list_all_documents()
    #     print(f"\n总计文档数量: {len(all_docs)}")

    #     # 按来源分组统计
    #     source_count = {}
    #     for doc in all_docs:
    #         source = doc.metadata.get('source', 'unknown')
    #         source_count[source] = source_count.get(source, 0) + 1

    #     print("\n按来源统计:")
    #     for source, count in source_count.items():
    #         print(f"  {source}: {count} 个文档")

    # else:
    #     print("向量存储未初始化或为空，请先添加文档")

    # 测试向量标准化检查
    # print("="*100)
    # print("检查向量标准化状态")
    # print("="*100)

    # vector_store = VectorStore()

    # if vector_store.is_ready():
    #     # 检查第一个文档的向量标准化状态
    #     print("检查第一个文档的向量标准化状态:")
    #     vector_store.print_vector_analysis(doc_index=0)

    #     print("\n" + "="*50)
    #     print("检查第二个文档的向量标准化状态:")
    #     vector_store.print_vector_analysis(doc_index=1)

    #     print("\n" + "="*50)
    #     print("检查第三个文档的向量标准化状态:")
    #     vector_store.print_vector_analysis(doc_index=2)

    #     # 批量检查多个向量的标准化状态
    #     print("\n" + "="*50)
    #     print("批量检查前10个向量的标准化状态:")
    #     normalized_count = 0
    #     total_checked = 10

    #     for i in range(min(total_checked, 10)):
    #         try:
    #             analysis = vector_store.check_vector_normalization(doc_index=i)
    #             if analysis['is_normalized']:
    #                 normalized_count += 1
    #             print(f"文档 {i+1}: L2范数={analysis['l2_norm']:.6f}, 标准化={'是' if analysis['is_normalized'] else '否'}")
    #         except Exception as e:
    #             print(f"文档 {i+1}: 检查失败 - {e}")

    #     print(f"\n标准化统计: {normalized_count}/{total_checked} 个向量已标准化")

    # else:
    #     print("向量存储未初始化或为空，请先添加文档")

    # 测试向量存储的索引类型
    # print("="*100)
    # print("测试向量存储的索引类型")
    # print("="*100)
    # vector_store = VectorStore()
    # print(vector_store.get_index_type())
    # print(vector_store.print_index_info())
    # print("="*100)
    # print("测试向量存储的索引类型")

    # 凡人修仙传入库
    print("="*100)
    print("凡人修仙传入库")
    print("="*100)
    from .text_processor import TextProcessor
    from .file_parser import FileParser
    text = FileParser.parse_txt_raw("examples/凡人修仙传test.txt")
    text_processor = TextProcessor()
    documents = text_processor.long_text_novel_split(text)
    vector_store = VectorStore()
    vector_store.add_documents(documents, batch_size=50)
    print(vector_store.get_vector_store_info())

    ...
