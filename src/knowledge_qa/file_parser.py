"""文件解析"""

import re
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)


class TextFileParser:
    """文本文件解析器"""

    @staticmethod
    def strip_text(text: str) -> str:
        """清理多余换行、空格、制表符等"""
        # 将多个连续空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 清理首尾空白
        text = text.strip()
        return text

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """解析PDF文件"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        return TextFileParser.strip_text(text)

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """解析DOCX文件"""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        return TextFileParser.strip_text(text)

    @staticmethod
    def parse_txt(file_path: str) -> str:
        """解析TXT文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return TextFileParser.strip_text(text)

    @staticmethod
    def parse_md(file_path: str) -> str:
        """解析MD文件"""
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        return TextFileParser.strip_text(text)

    # @staticmethod
    # def parse_csv(file_path: str) -> str:
    #     """解析CSV文件"""
    #     loader = CSVLoader(file_path)
    #     docs = loader.load()
    #     return "\n".join([doc.page_content for doc in docs])

    @staticmethod
    def parse_file(file_path: str) -> str:
        """解析文件"""
        if file_path.endswith('.pdf'):
            return TextFileParser.parse_pdf(file_path)
        elif file_path.endswith('.docx'):
            return TextFileParser.parse_docx(file_path)
        elif file_path.endswith('.txt'):
            return TextFileParser.parse_txt(file_path)
        elif file_path.endswith('.md'):
            return TextFileParser.parse_md(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.file_parser
    txt = TextFileParser.parse_file("examples/三国演义.txt")
    print("txt文件解析测试：", "\n", txt[:1000], "\n")
    pdf = TextFileParser.parse_file("examples/简历_3页.pdf")
    print("pdf文件解析测试：", "\n", pdf[:1000], "\n")
    docx = TextFileParser.parse_file("examples/报告_1页.docx")
    print("docx文件解析测试：", "\n", docx[:1000], "\n")
    md = TextFileParser.parse_file("examples/简历_3页.md")
    print("md文件解析测试：", "\n", md[:1000], "\n")
