"""文本分段处理器"""

from typing import List
import re
from difflib import SequenceMatcher
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

    def long_text_novel_split(self, text: str) -> List[Document]:
        """长文本小说分段，按照章节进行分段（增强版，处理重复或相似标题）"""
        if not text or not text.strip():
            log.warning("⚠️ 输入文本为空")
            return []

        import re

        # 定义章节匹配模式
        chapter_patterns = [
            r'第[一二三四五六七八九十百千万\d]+章.*',  
            r'第[一二三四五六七八九十百千万\d]+回.*',
            r'第[一二三四五六七八九十百千万\d]+节.*',
            r'第[一二三四五六七八九十百千万\d]+集.*',
            r'第[一二三四五六七八九十百千万\d]+部.*',
            r'第[一二三四五六七八九十百千万\d]+卷.*',
            r'第[一二三四五六七八九十百千万\d]+篇.*',
            r'第[一二三四五六七八九十百千万\d]+册.*',
            r'第[一二三四五六七八九十百千万\d]+本.*',
            r'第[一二三四五六七八九十百千万\d]+话.*',
        ]

        combined_pattern = '|'.join(chapter_patterns)

        # 查找所有章节位置
        chapters = [(m.start(), m.group().strip()) for m in re.finditer(combined_pattern, text, re.MULTILINE)]

        if not chapters:
            log.warning("⚠️ 未找到章节标题，使用默认分段")
            return self.split_text(text)

        # 去重相似标题，避免重复章节截断正文
        filtered_chapters = []
        for start_pos, chapter_title in chapters:
            if filtered_chapters:
                prev_title = filtered_chapters[-1][1]
                ratio = SequenceMatcher(None, prev_title, chapter_title).ratio()
                if ratio > 0.8:  # 相似度超过 80% 就认为重复
                    continue
            filtered_chapters.append((start_pos, chapter_title))

        # 创建文档列表
        documents = []
        for i, (start_pos, chapter_title) in enumerate(filtered_chapters):
            end_pos = filtered_chapters[i + 1][0] if i + 1 < len(filtered_chapters) else len(text)
            chapter_content = text[start_pos:end_pos].strip()
            if chapter_content:
                doc = Document(
                    page_content=chapter_content,
                    metadata={
                        "chapter_title": chapter_title,
                        "chapter_number": i + 1,
                        "source": "novel_split",
                        "content_type": "chapter"
                    }
                )
                documents.append(doc)

        log.info(f"📚 小说按章节分段完成，共 {len(documents)} 个章节")
        return documents


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.text_processor
    from .file_parser import FileParser

    txt_path = "examples/凡人修仙传test.txt"
    txt = FileParser.parse_txt_raw(txt_path)
    text_processor = TextProcessor()
    documents = text_processor.long_text_novel_split(txt)
    for doc in documents[:20]:
        print(doc.metadata)
        print(doc.page_content[:100], len(doc.page_content))
        print("-"*100)

    # """测试文本分段处理器"""
    # print("="*100)
    # print("开始测试文本分段处理器")
    # print("="*100)
    
    # from .file_parser import FileParser
    
    # # 步骤1: 解析文件
    # print("\n📖 步骤1: 解析文件")
    # text = FileParser.parse_file("examples/三国演义.txt")
    # print(f"文件解析成功，文本长度: {len(text)} 字符")
    # print(f"文本预览: {text[:100]}...\n")
    
    # # 步骤2: 初始化分段器并分段
    # print("📝 步骤2: 初始化分段器并分段")
    # text_processor = TextProcessor()
    
    # # 显示分段器配置信息
    # splitter_info = text_processor.get_splitter_info()
    # print(f"分段器配置:")
    # for key, value in splitter_info.items():
    #     print(f"   {key}: {value}")
    
    # documents: List[Document] = text_processor.split_text(text)
    # print(f"\n文档已分段，共 {len(documents)} 段")
    
    # # 显示前3段预览
    # print(f"\n📋 前3段预览：")
    # for i, doc in enumerate(documents[:3], 1):
    #     print(f"\n第 {i} 段:")
    #     print(doc.page_content[:200] + "...")
    #     print("-"*100)
    
    # # 步骤3: 测试文档分段功能
    # print("\n📚 步骤3: 测试文档分段功能")
    # test_docs = documents[:5]  # 取前5个文档进行测试
    # split_docs = text_processor.split_documents(test_docs)
    # print(f"原始文档数: {len(test_docs)}")
    # print(f"分段后文档数: {len(split_docs)}")
    
    # # 步骤4: 测试不同配置的分段器
    # print("\n⚙️ 步骤4: 测试不同配置的分段器")
    # custom_processor = TextProcessor(chunk_size=200, chunk_overlap=50)
    # custom_docs = custom_processor.split_text(text[:1000])  # 只测试前1000字符
    # print(f"自定义配置分段结果: {len(custom_docs)} 段")
    # print(f"自定义配置: chunk_size=200, chunk_overlap=50")
    
    # print("\n" + "="*100)
    # print("✅ 文本分段处理器测试完成!")
    # print("="*100)
