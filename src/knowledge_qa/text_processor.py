"""文本分段处理器"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .log_manager import log


class TextProcessor:
    """文本分段处理器"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        """初始化文本分段器"""
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # 创建分段器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self, text: str) -> List[Document]:
        """分段文本"""
        if not text or not text.strip():
            log.warning("⚠️ 输入文本为空")
            return []
        
        documents = self.splitter.create_documents([text])
        log.info(f"📝 文本分段完成，共 {len(documents)} 段")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分段文档列表"""
        if not documents:
            log.warning("⚠️ 输入文档列表为空")
            return []
        
        split_docs = self.splitter.split_documents(documents)
        log.info(f"📝 文档分段完成，从 {len(documents)} 个文档分段为 {len(split_docs)} 段")
        return split_docs
    
    def get_splitter_info(self) -> dict:
        """获取分段器信息"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.splitter._separators,
            "is_recursive": hasattr(self.splitter, '_separators')
        }


if __name__ == "__main__":
    """测试文本分段处理器"""
    print("="*100)
    print("开始测试文本分段处理器")
    print("="*100)
    
    from .file_parser import TextFileParser
    
    # 步骤1: 解析文件
    print("\n📖 步骤1: 解析文件")
    text = TextFileParser.parse_file("examples/三国演义.txt")
    print(f"文件解析成功，文本长度: {len(text)} 字符")
    print(f"文本预览: {text[:100]}...\n")
    
    # 步骤2: 初始化分段器并分段
    print("📝 步骤2: 初始化分段器并分段")
    text_processor = TextProcessor()
    
    # 显示分段器配置信息
    splitter_info = text_processor.get_splitter_info()
    print(f"分段器配置:")
    for key, value in splitter_info.items():
        print(f"   {key}: {value}")
    
    documents: List[Document] = text_processor.split_text(text)
    print(f"\n文档已分段，共 {len(documents)} 段")
    
    # 显示前3段预览
    print(f"\n📋 前3段预览：")
    for i, doc in enumerate(documents[:3], 1):
        print(f"\n第 {i} 段:")
        print(doc.page_content[:200] + "...")
        print("-"*100)
    
    # 步骤3: 测试文档分段功能
    print("\n📚 步骤3: 测试文档分段功能")
    test_docs = documents[:5]  # 取前5个文档进行测试
    split_docs = text_processor.split_documents(test_docs)
    print(f"原始文档数: {len(test_docs)}")
    print(f"分段后文档数: {len(split_docs)}")
    
    # 步骤4: 测试不同配置的分段器
    print("\n⚙️ 步骤4: 测试不同配置的分段器")
    custom_processor = TextProcessor(chunk_size=200, chunk_overlap=50)
    custom_docs = custom_processor.split_text(text[:1000])  # 只测试前1000字符
    print(f"自定义配置分段结果: {len(custom_docs)} 段")
    print(f"自定义配置: chunk_size=200, chunk_overlap=50")
    
    print("\n" + "="*100)
    print("✅ 文本分段处理器测试完成!")
    print("="*100)
