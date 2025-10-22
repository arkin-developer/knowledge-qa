""" 阅读本地文件资料的工具模型"""

import os
from tkinter import N
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langsmith import traceable

from ..config import settings
from ..log_manager import log
from ..memory import MemoryManager


@dataclass
class DocumentFragment:
    """文档片段"""
    filename: str
    content: str
    start_line: int
    end_line: int

@dataclass
class ReaderResult:
    """搜索结果"""
    fragments: List[DocumentFragment]

class ReaderLLM:
    """阅读本地文件资料的工具模型"""

    def __init__(self, files: str = os.path.join("/Users/arkin/Desktop/Dev/knowledge-qa/temp")):
        self.files = files
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ).bind_tools(self.get_tools()).with_structured_output(ReaderResult)

        self.memory = MemoryManager()
        log.info(f"初始化ReaderLLM模型: {settings.llm_model}")

    def get_tools(self) -> List:
        """获取所有工具函数"""
        return [
            self.list_files_tool,
            self.read_file_content_tool,
            self.search_in_file_tool,
            self.find_relevant_content_tool
        ]

    @tool
    def list_files_tool(self) -> str:
        """列出资料文件夹中的所有文件"""
        try:
            if not os.path.exists(self.files):
                return f"文件夹不存在: {self.files}"
            
            files = []
            for filename in os.listdir(self.files):
                file_path = os.path.join(self.files, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        "name": filename,
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
            
            result = "可用文件列表:\n"
            for file_info in files:
                result += f"- {file_info['name']} (大小: {file_info['size']} 字节)\n"
            
            return result
        except Exception as e:
            log.error(f"列出文件失败: {e}")
            return f"列出文件失败: {e}"

    @tool
    def read_file_content_tool(self, filename: str, start_line: int = 1, end_line: int = None) -> str:
        """读取文件内容，支持指定行数范围"""
        try:
            file_path = os.path.join(self.files, filename)
            if not os.path.exists(file_path):
                return f"文件不存在: {filename}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, end_line) if end_line else total_lines
            
            content = ''.join(lines[start_idx:end_idx])
            
            result = f"文件: {filename}\n"
            result += f"总行数: {total_lines}\n"
            result += f"读取行数: {start_line}-{end_idx}\n"
            result += f"内容:\n{content}"
            
            return result
        except Exception as e:
            log.error(f"读取文件失败: {e}")
            return f"读取文件失败: {e}"

    @tool
    def search_in_file_tool(self, filename: str, keyword: str, context_lines: int = 3) -> str:
        """在文件中搜索关键词，返回匹配的上下文"""
        try:
            file_path = os.path.join(self.files, filename)
            if not os.path.exists(file_path):
                return f"文件不存在: {filename}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            matches = []
            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    start_idx = max(0, i - context_lines)
                    end_idx = min(len(lines), i + context_lines + 1)
                    context = ''.join(lines[start_idx:end_idx])
                    matches.append({
                        "line_number": i + 1,
                        "context": context
                    })
            
            if not matches:
                return f"在文件 {filename} 中未找到关键词: {keyword}"
            
            result = f"在文件 {filename} 中找到 {len(matches)} 个匹配项:\n"
            for match in matches[:10]:  # 限制显示前10个匹配
                result += f"第 {match['line_number']} 行:\n{match['context']}\n---\n"
            
            return result
        except Exception as e:
            log.error(f"搜索文件失败: {e}")
            return f"搜索文件失败: {e}"

    @tool
    def find_relevant_content_tool(self, filename: str, query: str, max_results: int = 5) -> str:
        """根据查询条件智能定位相关内容"""
        try:
            file_path = os.path.join(self.files, filename)
            if not os.path.exists(file_path):
                return f"文件不存在: {filename}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的关键词匹配和相关性评分
            query_words = query.lower().split()
            lines = content.split('\n')
            
            scored_lines = []
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                line_lower = line.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score > 0:
                    scored_lines.append((score, i + 1, line))
            
            # 按相关性排序
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            
            if not scored_lines:
                return f"在文件 {filename} 中未找到相关内容"
            
            result = f"在文件 {filename} 中找到相关内容:\n"
            for score, line_num, line in scored_lines[:max_results]:
                result += f"第 {line_num} 行 (相关性: {score}): {line.strip()}\n"
            
            return result
        except Exception as e:
            log.error(f"查找相关内容失败: {e}")
            return f"查找相关内容失败: {e}"


    def get_system_prompt(self) -> str:
        """获取提示词"""
        return """
你是一个专门查询资料的助手，用户输入问题，你根据问题去本地文件去查询资料，直到查询到关键信息为止。

你有以下工具可以使用：
1. list_files_tool: 列出资料文件夹中的所有文件(无参数)
2. read_file_content_tool: 读取文件内容，支持指定行数范围(start_line, end_line)
3. search_in_file_tool: 在文件中搜索关键词，返回匹配的上下文(filename, keyword, context_lines)  
4. find_relevant_content_tool: 根据查询条件智能定位相关内容(filename, query, max_results)

使用策略：
1. 首先使用 list_files_tool 查看有哪些文件可用
2. 根据用户问题，选择合适的文件进行查询
3. 使用 search_in_file_tool 或 find_relevant_content_tool 定位相关内容
4. 如果需要详细内容，使用 read_file_content_tool 读取具体内容
5. 你可以根据查询记录去决策下一步的动作，例如查询行数范围是否需要调整，或者使用其他工具来获取更多信息
6. 每次对片段的整理必须符合DocumentFragment的格式，并返回给用户

重要：你必须主动使用这些工具来查找相关原文，然后将找到的原文（精准匹配原文）整理返回给用户。

注意：最终输出的时候是多个片段，格式为
ReaderResult 格式：{
    "fragments": [DocumentFragment]
}
DocumentFragment格式：
{
    "filename": "文件名",
    "content": "文件内容",
    "start_line": "起始行数",
    "end_line": "结束行数"
}
        """

    def generate(self, query: str = None, suggestions: str = None, use_memory: bool = True) -> ReaderResult:
        """根据问题生成回答"""
        if query:
            # 第一轮资料查询
            user_prompt = f"用户的问题：{query}"
        else:
            # 如果下一个节点认为没有查询到就继续根据建议查询资料
            user_prompt = f"经过验证，用户需要二次检索，建议：{suggestions}"

        try:
            messages: List[BaseMessage] = [
                SystemMessage(content=self.get_system_prompt()),
            ]

            if use_memory and self.memory:
                history_messages = self.memory.get_messages()
                if history_messages:
                    messages.extend(history_messages)
                    log.debug(f"加载历史对话: {len(history_messages)} 条消息")

            messages.append(HumanMessage(content=user_prompt))

            response = self.llm.invoke(messages)

            if use_memory and self.memory:
                # 处理不同格式的response
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                self.memory.add_exchange(user_prompt, content)
                log.debug("已保存本轮对话到记忆")

            # 解析返回的ReaderResult
            if isinstance(response, ReaderResult):
                return response
            elif isinstance(response, dict) and "fragments" in response:
                # 如果返回的是字典格式，解析为ReaderResult
                fragments_data = response.get("fragments", [])
                fragments = []
                for frag_data in fragments_data:
                    fragment = DocumentFragment(
                        filename=frag_data.get("filename", ""),
                        content=frag_data.get("content", ""),
                        start_line=frag_data.get("start_line", 0),
                        end_line=frag_data.get("end_line", 0)
                    )
                    fragments.append(fragment)
                return ReaderResult(fragments=fragments)
            else:
                # 如果返回的是其他格式，返回空的ReaderResult
                return ReaderResult(fragments=[])

        except Exception as e:
            log.error(f"回答问题失败: {e}")
            raise e

    def clear_memory(self) -> None:
        """清空对话记忆"""
        if self.memory:
            self.memory.clear()
            log.info("已清空ReaderLLM记忆")
