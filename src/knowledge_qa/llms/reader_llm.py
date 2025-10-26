"""阅读本地文件资料的工具智能体"""

import os 
import json
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
from ..config import settings
from ..log_manager import log


class DocumentFragmentMeta(BaseModel):
    """文档片段元数据（不含内容，节省token）"""
    filename: str
    start_line: int
    end_line: int

class DocumentFragment(BaseModel):
    """文档片段"""
    filename: str
    content: str
    start_line: int
    end_line: int

class SearchKeywordToolInput(BaseModel):
    keyword: str = Field(..., description="关键词")
    filename: str = Field(..., description="文件名")
    limit: int = Field(default=30, description="最大返回结果数")
    
    @classmethod
    def get_example_format(cls) -> str:
        """获取参数格式示例"""
        return '{"keyword": "关键词", "filename": "文件名", "limit": 100}'
    
    @classmethod
    def get_schema_dict(cls) -> dict:
        """获取参数格式的字典表示"""
        return {
            "keyword": "关键词 (必填)",
            "filename": "文件名 (必填)", 
            "limit": "最大返回结果数 (可选，默认100)"
        }

class ReadFileContentToolInput(BaseModel):
    filename: str = Field(..., description="文件名")
    start_line: int = Field(..., description="起始行号")
    end_line: int = Field(..., description="结束行号")
    
    @classmethod
    def get_example_format(cls) -> str:
        """获取参数格式示例"""
        return '{"filename": "文件名", "start_line": 1, "end_line": 10}'
    
    @classmethod
    def get_schema_dict(cls) -> dict:
        """获取参数格式的字典表示"""
        return {
            "filename": "文件名 (必填)",
            "start_line": "起始行号 (必填)",
            "end_line": "结束行号 (必填)"
        }

class AddFragmentMetaToolInput(BaseModel):
    filename: str = Field(..., description="文件名")
    start_line: int = Field(..., description="起始行号")
    end_line: int = Field(..., description="结束行号")
    
    @classmethod
    def get_example_format(cls) -> str:
        """获取参数格式示例"""
        return '{"filename": "文件名", "start_line": 1, "end_line": 10}'
    
    @classmethod
    def get_schema_dict(cls) -> dict:
        """获取参数格式的字典表示"""
        return {
            "filename": "文件名 (必填)",
            "start_line": "起始行号 (必填)",
            "end_line": "结束行号 (必填)"
        }

class ReaderLLM:
    """阅读本地文件资料的工具智能体"""

    def __init__(self):
        self.upload_path = settings.upload_temp_path
        self.tools = self._create_all_tools()
        self.llm = ChatOpenAI(
            openai_api_key=settings.siliconcloud_api_key,
            openai_api_base=settings.siliconcloud_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        )

        # 文档片段元数据列表
        self.fragments_meta = []
        # 完整文档片段列表（编程获取，保证精准匹配）
        self.fragments = []
        
        # 创建日志记录回调（调试）
        self.agent_callback = AgentLoggingCallback()
        
        for tool in self.tools:
            log.info(f"  - {tool.name}: {tool.description}")

        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_prompt_template()
        )
        
        # 创建AgentExecutor来执行agent
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            callbacks=[self.agent_callback],
            max_iterations=10,
            early_stopping_method="force"
        )
        log.info("Agent 和 AgentExecutor 创建完成")
    
    def _get_prompt_template(self) -> PromptTemplate:
        """创建ReAct Agent的prompt模板（按照官方文档格式）,官方建议react agent 的prompt 用英文"""
        return PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

**Tool Usage Instructions:**
- list_files_tool_func(): List all available files, no parameters needed
- search_keyword_tool_func(keyword, filename, limit=300): Search for keywords in files and return relevant lines with context
  Purpose: Find specific content related to keywords, returns line numbers and surrounding context
  Parameters: keyword (search term - can be multiple keywords), filename (file name), limit (max results, default 300)
  Use this when: You need to find specific information or content within a file
  
  **KEYWORD SEPARATION RULES:**
  - **Chinese keywords**: Use spaces to separate multiple keywords
    Examples: "韩立 七玄门", "法术 修炼", "墨大夫 来历", "死契 血斗"
  - **English keywords**: Use COMMAS to separate multiple keywords (preserves phrases)
    Examples: "bonus action, spellcasting", "casting time, cantrip", "Grappled, status"
    Examples: "bonus action spell", "spellcasting rules" (single phrases)
  
  The function will find lines containing ANY of these keywords (OR logic)
- read_file_content_tool_func(filename, start_line, end_line): Read file content by line range
  Purpose: Get detailed content from specific line ranges, useful after finding relevant lines with search
  Parameters: filename (file name), start_line (start line number), end_line (end line number)
  Use this when: You need to read more detailed content after finding relevant lines with search_keyword_tool_func
- add_fragment_meta_tool_func(fragments): Add document fragment metadata to the system
  Purpose: Store relevant document fragments for future reference and context building
  Parameters: fragments (array of fragment objects with filename, start_line, end_line)
  Use this when: You find relevant content that should be saved for the user's question
  Note: You can call this multiple times and can input single or multiple fragments as an array
  **CRITICAL**: This is your PRIMARY GOAL - save ALL relevant fragments you discover
  Format examples:
    Single fragment: {{"filename": "file.txt", "start_line": 10, "end_line": 20}}
    Multiple fragments: {{"fragments": [{{"filename": "file.txt", "start_line": 10, "end_line": 20}}, {{"filename": "file.txt", "start_line": 30, "end_line": 40}}]}}
    
  **CRITICAL JSON FORMAT REQUIREMENTS**:
  - Use standard JSON format with proper punctuation
  - Numbers (start_line, end_line) must NOT have quotes around them
  - Use English punctuation only (no Chinese punctuation)
  - Use single-line JSON format to avoid parsing errors
  - Example: {{"filename": "file.txt", "start_line": 10, "end_line": 20}} ✅
  - Wrong: {{"filename": "file.txt", "start_line": "10", "end_line": "20"}} ❌ (numbers should not have quotes)
  - Wrong: {{"filename": "Player's Handbook.md", "start_line": 16845, "end_line": 16852"}} ❌ (extra quote after number)

**Workflow Guidelines:**
1. First, use list_files_tool_func() to see available files
2. Use search_keyword_tool_func() to find relevant content (returns lines containing keywords)
3. **EVALUATE CONTENT**: After finding relevant lines, check if the keyword-containing lines provide enough context:
   - **If context is sufficient**: Directly use add_fragment_meta_tool_func() to save the relevant fragments
   - **If context is insufficient**: Use read_file_content_tool_func() to get surrounding lines for more context
4. **CRITICAL**: Use add_fragment_meta_tool_func() to save ALL relevant fragments you find (this is your PRIMARY GOAL)
5. Always provide a complete answer based on the information you gather

**Key Understanding**: 
- search_keyword_tool_func() returns **complete lines** that contain the keywords
- read_file_content_tool_func() returns **surrounding context** around specific line ranges
- Only use read_file_content_tool_func() when the keyword-containing lines lack sufficient context to answer the question fully

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (use the correct parameter names as shown above)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**Important Notes:**
- **YOUR MAIN OBJECTIVE**: Find and save ALL relevant document fragments using add_fragment_meta_tool_func
- **SMART CONTEXT STRATEGY**: After finding relevant content with search_keyword_tool_func:
  * **Understand the difference**: search_keyword_tool_func() returns complete lines containing keywords, read_file_content_tool_func() returns surrounding context
  * **Evaluate first**: Check if the keyword-containing lines provide enough context to answer the question
  * **If sufficient**: Save the fragments directly using add_fragment_meta_tool_func()
  * **If insufficient**: Use read_file_content_tool_func() to get surrounding lines for more context
  * **Be strategic**: Only read file content when the keyword-containing lines lack sufficient context
- **SEARCH STRATEGY**:
  * **English keywords**: Use commas to separate multiple keywords (e.g., "bonus action, spellcasting", "casting time, cantrip")
  * **Chinese keywords**: Use spaces to separate multiple keywords (e.g., "韩立 七玄门", "法术 修炼")
  * **Mixed language**: Use appropriate separator for each language (e.g., "bonus action, 奖励动作")
  * **Single phrases**: Keep as one unit (e.g., "bonus action spell", "spellcasting rules")
- **MANDATORY**: Use add_fragment_meta_tool_func to save important fragments you discover (supports both single and multiple fragments)
- You can call add_fragment_meta_tool_func multiple times throughout your search process
- **PRIORITY**: Saving relevant fragments is more important than just answering the question
- Always follow the exact format: Thought -> Action -> Action Input -> Observation
- If you have enough information to answer the question, proceed to Final Answer
- Never skip the Action line - always include it when you want to use a tool

Begin!

Question: {input}
Thought:{agent_scratchpad}
        """)
    
    def generate(self, query: str) -> str:
        """根据问题生成回答"""
        log.info(f"开始处理查询: {query}")
        try:
            result = self.agent_executor.invoke({"input": query})
            log.info("查询处理完成")
            return result
        except Exception as e:
            log.error(f"查询处理失败: {e}")
            log.error(f"错误类型: {type(e)}")
            log.error(f"错误详情: {str(e)}")
            import traceback
            log.error(f"错误堆栈: {traceback.format_exc()}")
            raise

    def _create_all_tools(self) -> List[Tool]:
        """创建所有工具"""
        tools = [
            self._list_files_tool(),
            self._search_keyword_tool(),
            self._read_file_content_tool(),
            self._add_fragment_meta_tool()
        ]
        return tools

    def _list_files_tool(self):
        """列出文件工具"""
        @tool
        def list_files_tool_func() -> str:
            """列出上传文件夹中的所有文件"""
            try:
                log.info(f"正在检查目录: {self.upload_path}")
                if not os.path.exists(self.upload_path):
                    log.warning(f"目录不存在: {self.upload_path}")
                    return f"文件夹不存在: {self.upload_path}"

                files = []
                for filename in os.listdir(self.upload_path):
                    file_path = os.path.join(self.upload_path, filename)
                    if os.path.isfile(file_path):
                        files.append(filename)
                
                result = json.dumps({"files": files}, ensure_ascii=False, indent=2)
                log.info(f"找到文件: {result}")
                return result
            except Exception as e:
                log.error(f"列出文件失败: {e}")
                return f"列出文件失败: {e}"
        return list_files_tool_func

    def _search_keyword_tool(self):
        """搜索关键词相关内容工具"""
        @tool
        def search_keyword_tool_func(input: SearchKeywordToolInput | str) -> str:
            """在指定文件中搜索关键词相关内容"""
            # 参数解析
            try:
                if isinstance(input, str):
                    input = json.loads(input)
                    keyword = input.get("keyword")
                    filename = input.get("filename")
                    limit = input.get("limit")
                else:
                    keyword = input.keyword
                    filename = input.filename
                    limit = input.limit
            except Exception as e:
                log.error(f"搜索关键词失败: {e}")
                return f"输入参数有误，请参考格式: {SearchKeywordToolInput.get_example_format()}，重新检查后重试。"

            try:
                # 智能关键词分割：根据语言特点选择分割方式
                keywords = self._split_keywords(keyword)

                file_path = os.path.join(self.upload_path, filename)
                if not os.path.exists(file_path):
                    return f"{filename}文件不存在，请核实文件名称是否正确。"
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        return f"无法读取文件 {filename}，编码不支持"

                lines = content.split('\n')
                relevant_lines = []
                for i, line in enumerate(lines, 1):
                    if any(keyword.lower() in line.lower() for keyword in keywords):
                        # 扩大上下文范围，增加匹配概率
                        # content = lines[i-1].strip() + line.strip() + lines[i+1].strip() 
                        relevant_lines.append({
                            "line_number": i,
                            "content": line.strip()  
                        })
                if limit is None:
                    limit = 30  # 默认值
                if limit > 150:
                    limit = 150
                if len(relevant_lines) > limit:
                    relevant_lines = self._smart_sample_lines(relevant_lines, limit)
                result = {
                    "filename": filename,
                    "total_matches": len(relevant_lines),
                    "relevant_lines": relevant_lines
                }
                return json.dumps(result, ensure_ascii=False, indent=2)
                log.info(f"搜索关键词 {keyword} 在文件 {filename} 中找到 {len(relevant_lines)} 行相关内容，返回前 {len(relevant_lines)} 行")
            except Exception as e:
                log.error(f"搜索关键词失败: {e}")
                return f"搜索关键词失败: {e}"
        return search_keyword_tool_func

    def _fix_json_format(self, json_str: str) -> str:
        """修复JSON格式问题"""
        # 移除尾随逗号
        fixed_input = re.sub(r',\s*}', '}', json_str)
        fixed_input = re.sub(r',\s*]', ']', fixed_input)
        
        # 移除换行符和制表符
        fixed_input = re.sub(r'\n\s*', '', fixed_input)
        fixed_input = re.sub(r'\t', ' ', fixed_input)
        
        # 修复不闭合的双引号问题
        fixed_input = re.sub(r'(\d+)"\s*}', r'\1}', fixed_input)
        fixed_input = re.sub(r'(\d+)"\s*]', r'\1]', fixed_input)
        fixed_input = re.sub(r'(\d+)"\s*,', r'\1,', fixed_input)
        
        # 修复不匹配的括号
        fixed_input = self._fix_unmatched_brackets(fixed_input)
        
        return fixed_input.strip()
    
    def _fix_unmatched_brackets(self, json_str: str) -> str:
        """修复不匹配的括号"""
        # 统计括号数量
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 修复缺失的闭合括号
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str

    def _split_keywords(self, keyword_str: str) -> List[str]:
        """智能分割关键词，根据语言特点选择分割方式"""
        if not keyword_str:
            return []
        
        # 检测是否包含中文字符
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', keyword_str))
        
        if has_chinese:
            # 中文：使用空格和逗号分割
            keywords = re.split(r'[,\s]+', keyword_str)
        else:
            # 英文：优先使用逗号分割，如果没有逗号则使用空格
            if ',' in keyword_str:
                keywords = [k.strip() for k in keyword_str.split(',') if k.strip()]
            else:
                # 对于英文，保持短语完整性，只在明显分隔处分割
                keywords = re.split(r'\s+', keyword_str)
        
        # 过滤空字符串
        keywords = [k.strip() for k in keywords if k.strip()]
        
        return keywords

    def _smart_sample_lines(self, relevant_lines: List[Dict], limit: int) -> List[Dict]:
        """智能采样相关行，优先选择包含多个关键词的重要行"""
        if len(relevant_lines) <= limit:
            return relevant_lines
        
        # 按行号排序
        sorted_lines = sorted(relevant_lines, key=lambda x: x['line_number'])
        total_lines = len(sorted_lines)
        
        if total_lines <= limit:
            return sorted_lines
        
        # 计算每行的重要性分数（基于关键词密度和内容长度）
        def calculate_importance(line_info):
            content = line_info['content'].lower()
            # 计算关键词密度
            keyword_count = sum(1 for keyword in ['bonus action', 'spellcasting', 'cantrip', 'casting time', 'same turn'] 
                              if keyword in content)
            # 计算内容长度（避免过短的行）
            content_length = len(content)
            # 综合分数：关键词密度 + 内容长度权重
            return keyword_count * 10 + min(content_length / 10, 5)
        
        # 按重要性排序
        scored_lines = [(line, calculate_importance(line)) for line in sorted_lines]
        scored_lines.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前limit个最重要的行
        sampled_lines = [line for line, score in scored_lines[:limit]]
        
        # 按行号重新排序
        return sorted(sampled_lines, key=lambda x: x['line_number'])

    def read_file_by_lines(self, input: Dict[str, Any] | str) -> str:
        """读取指定文件的行号范围内的内容，返回字符串"""
        # 参数解析
        try:
            if isinstance(input, str):
                # 尝试解析JSON，失败则修复格式后重试
                try:
                    parsed_input = json.loads(input.strip())
                except json.JSONDecodeError:
                    # 修复JSON格式问题
                    fixed_input = self._fix_json_format(input)
                    parsed_input = json.loads(fixed_input)
                
                filename = parsed_input.get("filename")
                # 兼容 start_line/end_line 和 start_index/end_index
                start_line = parsed_input.get("start_line") or parsed_input.get("start_index")
                end_line = parsed_input.get("end_line") or parsed_input.get("end_index")
            elif isinstance(input, dict):
                filename = input.get("filename")
                # 兼容 start_line/end_line 和 start_index/end_index
                start_line = input.get("start_line") or input.get("start_index")
                end_line = input.get("end_line") or input.get("end_index")
                
        except Exception as e:
            log.error(f"读取文件内容失败: {e}")
            return f"输入参数有误，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if not filename or not start_line or not end_line:
            return f"输入参数有误，文件名、起始行号和结束行号不能为空，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if start_line > end_line:
            return f"输入参数有误，起始行号不能大于结束行号，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if start_line < 1:
            return f"输入参数有误，起始行号不能小于1，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if end_line < 1:
            return f"输入参数有误，结束行号不能小于1，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"

        try:
            file_path = os.path.join(self.upload_path, filename)
            if not os.path.exists(file_path):
                return f"{filename}文件不存在，请核实文件名称是否正确。"
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    return f"无法读取文件 {filename}，编码不支持"
            
            lines = content.split('\n')
            if end_line - start_line > 50:
                end_line = start_line + 50
            selected_lines = lines[start_line-1:end_line]
            return json.dumps({"content": selected_lines}, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"读取文件内容失败: {e}")
            return f"读取文件内容失败: {e}"

    def _read_file_content_tool(self):
        """读取文件内容工具"""
        @tool
        def read_file_content_tool_func(input: Dict[str, Any] | str) -> str:
            """读取指定文件的行号范围内的内容"""
            return self.read_file_by_lines(input)
        return read_file_content_tool_func
    

    def _add_fragment_meta_tool(self):
        """添加文档片段元数据工具"""
        @tool
        def add_fragment_meta_tool_func(input: List[AddFragmentMetaToolInput] | AddFragmentMetaToolInput | str) -> str:
            """添加文档片段元数据"""
            try:
                fragment_meta_list: List[DocumentFragmentMeta] = []
                # 解析输入参数
                if isinstance(input, str):
                    try:
                        parsed_input = json.loads(input.strip())
                    except json.JSONDecodeError:
                        # 修复JSON格式问题
                        fixed_input = self._fix_json_format(input)
                        parsed_input = json.loads(fixed_input)
                    
                    # 处理解析后的数据
                    if "fragments" in parsed_input and isinstance(parsed_input["fragments"], list):
                        for item in parsed_input["fragments"]:
                            fragment_meta_list.append(DocumentFragmentMeta(
                                filename=item.get("filename"), 
                                start_line=item.get("start_line"), 
                                end_line=item.get("end_line")
                            ))
                    else:
                        fragment_meta_list.append(DocumentFragmentMeta(
                            filename=parsed_input.get("filename"), 
                            start_line=parsed_input.get("start_line"), 
                            end_line=parsed_input.get("end_line")
                        ))
                        
                elif isinstance(input, AddFragmentMetaToolInput):
                    fragment_meta_list.append(DocumentFragmentMeta(
                        filename=input.filename, 
                        start_line=input.start_line, 
                        end_line=input.end_line
                    ))
                elif isinstance(input, list):
                    for item in input:
                        if isinstance(item, AddFragmentMetaToolInput):
                            fragment_meta_list.append(DocumentFragmentMeta(
                                filename=item.filename, 
                                start_line=item.start_line, 
                                end_line=item.end_line
                            ))
                        elif isinstance(item, dict):
                            fragment_meta_list.append(DocumentFragmentMeta(
                                filename=item.get("filename"), 
                                start_line=item.get("start_line"), 
                                end_line=item.get("end_line")
                            ))
                
                # 验证参数
                for fragment in fragment_meta_list:
                    if not fragment.filename or not fragment.start_line or not fragment.end_line:
                        return f"输入参数有误，文件名、起始行号和结束行号不能为空，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
                    if fragment.start_line > fragment.end_line:
                        return f"输入参数有误，起始行号不能大于结束行号，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
                    if fragment.start_line < 1 or fragment.end_line < 1:
                        return f"输入参数有误，行号不能小于1，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"

                # 保存片段
                self.fragments_meta.extend(fragment_meta_list)
                log.info(f"添加文档片段元数据成功: {len(fragment_meta_list)} 个片段")
                return f"添加文档片段元数据成功，共保存 {len(fragment_meta_list)} 个片段"
                
            except Exception as e:
                log.error(f"添加文档片段元数据失败: {e}")
                return f"输入参数有误，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
        return add_fragment_meta_tool_func

    def clear_fragments_meta(self):
        """清空文档片段元数据列表"""
        self.fragments_meta = []

    def get_fragments_meta(self) -> List[DocumentFragmentMeta]:
        """获取文档片段元数据列表"""
        return self.fragments_meta

    def update_fragments(self):
        """更新文档片段列表"""
        self.fragments = [] # 清空旧的文档片段列表
        try:
            for fragment_meta in self.fragments_meta:
                content = self.read_file_by_lines({
                    "filename": fragment_meta.filename,
                    "start_line": fragment_meta.start_line,
                    "end_line": fragment_meta.end_line
                })
                if content:
                    self.fragments.append(DocumentFragment(
                        filename=fragment_meta.filename,
                        content=content,
                        start_line=fragment_meta.start_line,
                        end_line=fragment_meta.end_line
                    ))
        except Exception as e:
            log.error(f"更新文档片段列表失败: {e}")
            return f"更新文档片段列表失败: {e}"

    def get_fragments(self) -> List[DocumentFragment]:
        """获取文档片段列表"""
        return self.fragments

class AgentLoggingCallback(BaseCallbackHandler):
    """Agent执行过程的详细日志记录回调"""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """记录LLM开始处理"""
        log.info(f"🧠 LLM Start: 处理 {len(prompts)} 个提示")
        for i, prompt in enumerate(prompts):
            log.info(f"  提示 {i+1}: {prompt[:200]}...")
    
    def on_llm_end(self, response, **kwargs):
        """记录LLM处理完成"""
        log.info(f"🧠 LLM End: {response.generations[0][0].text[:200]}...")
    
    def on_llm_error(self, error, **kwargs):
        """记录LLM错误"""
        log.error(f"❌ LLM Error: {error}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """记录Chain开始"""
        if serialized:
            log.info(f"🔗 Chain Start: {serialized.get('name', 'Unknown')}")
        else:
            log.info(f"🔗 Chain Start: Unknown")
        log.info(f"  输入: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        """记录Chain结束"""
        log.info(f"🔗 Chain End: {outputs}")
    
    def on_chain_error(self, error, **kwargs):
        """记录Chain错误"""
        log.error(f"❌ Chain Error: {error}")
    
    def on_agent_action(self, action, **kwargs):
        """记录Agent执行的动作"""
        log.info(f"🤖 Agent Action: {action.tool}")
        log.info(f"  工具输入: {action.tool_input}")
        log.info(f"  日志: {action.log}")
    
    def on_agent_finish(self, finish, **kwargs):
        """记录Agent完成执行"""
        log.info(f"✅ Agent Finish: {finish.return_values}")
        log.info(f"  日志: {finish.log}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """记录工具开始执行"""
        tool_name = serialized.get('name', 'Unknown')
        log.info(f"🔧 Tool Start: {tool_name}")
        log.info(f"  工具输入: {input_str}")
        log.info(f"  工具类型: {type(serialized)}")
        log.info(f"  序列化数据: {serialized}")
    
    def on_tool_end(self, output, **kwargs):
        """记录工具执行完成"""
        log.info(f"✅ Tool End: {output}")
    
    def on_tool_error(self, error, **kwargs):
        """记录工具执行错误"""
        log.error(f"❌ Tool Error: {error}")
        log.error(f"  错误类型: {type(error)}")
        log.error(f"  错误详情: {str(error)}")
    
    def on_chain_error(self, error, **kwargs):
        """记录Chain错误"""
        log.error(f"❌ Chain Error: {error}")
        log.error(f"  错误类型: {type(error)}")
        log.error(f"  错误详情: {str(error)}")
    
    def on_llm_error(self, error, **kwargs):
        """记录LLM错误"""
        log.error(f"❌ LLM Error: {error}")
        log.error(f"  错误类型: {type(error)}")
        log.error(f"  错误详情: {str(error)}")
    
    def on_text(self, text, **kwargs):
        """记录文本输出"""
        log.info(f"📝 Text: {text}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """记录LLM开始处理"""
        log.info(f"🧠 LLM Start: 处理 {len(prompts)} 个提示")
        for i, prompt in enumerate(prompts):
            log.info(f"  提示 {i+1}: {prompt[:200]}...")
    
    def on_llm_end(self, response, **kwargs):
        """记录LLM处理完成"""
        if response.generations and response.generations[0]:
            log.info(f"🧠 LLM End: {response.generations[0][0].text[:200]}...")
        else:
            log.info(f"🧠 LLM End: 无响应内容")
    
    def on_llm_error(self, error, **kwargs):
        """记录LLM错误"""
        log.error(f"❌ LLM Error: {error}")
        log.error(f"  错误类型: {type(error)}")
        log.error(f"  错误详情: {str(error)}")


# 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.llms.reader_llm
if __name__ == "__main__":
    reader_llm = ReaderLLM()
    # result = reader_llm.generate("韩立是如何进入七玄门的？记名弟子初次考验包含哪些关键路段与环节？")
    result = reader_llm.generate("当一个目标处于 Grappled（被擒抱/缠住）状态时，会发生什么？列出该状态对目标的具体机械效果。")
    print(result)
    print("=" * 50)
    print("文档片段元数据列表:")
    reader_llm.update_fragments()
    print(reader_llm.get_fragments())