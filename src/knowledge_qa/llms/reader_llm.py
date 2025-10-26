"""阅读本地文件资料的工具模型"""

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
    limit: int = Field(default=300, description="最大返回结果数")
    
    @classmethod
    def get_example_format(cls) -> str:
        """获取参数格式示例"""
        return '{"keyword": "关键词", "filename": "文件名", "limit": 300}'
    
    @classmethod
    def get_schema_dict(cls) -> dict:
        """获取参数格式的字典表示"""
        return {
            "keyword": "关键词 (必填)",
            "filename": "文件名 (必填)", 
            "limit": "最大返回结果数 (可选，默认300)"
        }

class ReadFileContentToolInput(BaseModel):
    filename: str = Field(..., description="文件名")
    start_index: int = Field(..., description="起始行号")
    end_index: int = Field(..., description="结束行号")
    
    @classmethod
    def get_example_format(cls) -> str:
        """获取参数格式示例"""
        return '{"filename": "文件名", "start_index": 1, "end_index": 10}'
    
    @classmethod
    def get_schema_dict(cls) -> dict:
        """获取参数格式的字典表示"""
        return {
            "filename": "文件名 (必填)",
            "start_index": "起始行号 (必填)",
            "end_index": "结束行号 (必填)"
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
    """阅读本地文件资料的工具模型"""

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
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.agent_callback],
            max_iterations=10,
            early_stopping_method="generate"
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
  Parameters: keyword (search term), filename (file name), limit (max results, default 300)
  Use this when: You need to find specific information or content within a file
- read_file_content_tool_func(filename, start_index, end_index): Read file content by line range
  Purpose: Get detailed content from specific line ranges, useful after finding relevant lines with search
  Parameters: filename (file name), start_index (start line number), end_index (end line number)
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

**Workflow Guidelines:**
1. First, use list_files_tool_func() to see available files
2. Then use search_keyword_tool_func() to find relevant content
3. If you need more context, use read_file_content_tool_func() to read specific line ranges
4. **CRITICAL**: Use add_fragment_meta_tool_func() to save ALL relevant fragments you find (this is your PRIMARY GOAL)
5. Always provide a complete answer based on the information you gather

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
- After using search_keyword_tool_func, if you find relevant content, you can use read_file_content_tool_func to get more context
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
        log.info(f"创建了 {len(tools)} 个工具")
        for i, tool in enumerate(tools):
            log.info(f"  工具 {i+1}: {tool.name}")
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
                # keyword 切分, 这里用正则表达式切分，空格，逗号常见的分隔符
                keywords = re.split(r'[,\s]+', keyword)

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
                        relevant_lines.append({
                            "line_number": i,
                            "content": line.strip()
                        })
                if limit is None:
                    limit = 300  # 默认值
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

    def _smart_sample_lines(self, relevant_lines: List[Dict], limit: int) -> List[Dict]:
        """智能采样相关行，确保覆盖全文的不同部分"""
        if limit is None:
            limit = 300  # 默认值
        if len(relevant_lines) <= limit:
            return relevant_lines
        
        # 按行号排序
        sorted_lines = sorted(relevant_lines, key=lambda x: x['line_number'])
        total_lines = len(sorted_lines)
        
        if total_lines <= limit:
            return sorted_lines
        
        # 计算采样间隔，确保覆盖全文
        step = total_lines / limit
        
        sampled_lines = []
        for i in range(limit):
            # 计算采样位置，确保均匀分布
            index = int(i * step)
            if index < total_lines:
                sampled_lines.append(sorted_lines[index])
        
        # 确保包含开头和结尾的重要内容
        if sorted_lines[0] not in sampled_lines:
            sampled_lines[0] = sorted_lines[0]
        if sorted_lines[-1] not in sampled_lines:
            sampled_lines[-1] = sorted_lines[-1]
        
        # 按行号重新排序
        return sorted(sampled_lines, key=lambda x: x['line_number'])

    def read_file_by_lines(self, input: Dict[str, Any] | str) -> str:
        """读取指定文件的行号范围内的内容，返回字符串"""
        # 参数解析
        try:
            if isinstance(input, str):
                # 清理输入字符串，移除可能的不可见字符
                cleaned_input = input.strip()
                input = json.loads(cleaned_input)
                filename = input.get("filename")
                # 兼容 start_line/end_line 和 start_index/end_index
                start_index = input.get("start_index") or input.get("start_line")
                end_index = input.get("end_index") or input.get("end_line")
            elif isinstance(input, dict):
                filename = input.get("filename")
                # 兼容 start_line/end_line 和 start_index/end_index
                start_index = input.get("start_index") or input.get("start_line")
                end_index = input.get("end_index") or input.get("end_line")
            else:
                filename = input.filename
                # 兼容 start_line/end_line 和 start_index/end_index
                start_index = getattr(input, 'start_index', None) or getattr(input, 'start_line', None)
                end_index = getattr(input, 'end_index', None) or getattr(input, 'end_line', None)
        except Exception as e:
            log.error(f"读取文件内容失败: {e}")
            log.error(f"输入参数: {repr(input)}")
            log.error(f"输入类型: {type(input)}")
            return f"输入参数有误，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if not filename or not start_index or not end_index:
            return f"输入参数有误，文件名、起始行号和结束行号不能为空，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if start_index > end_index:
            return f"输入参数有误，起始行号不能大于结束行号，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if start_index < 1:
            return f"输入参数有误，起始行号不能小于1，请参考格式: {ReadFileContentToolInput.get_example_format()}，重新检查后重试。"
        if end_index < 1:
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
            return json.dumps({"content": lines[start_index:end_index]}, ensure_ascii=False, indent=2)
            log.info(f"读取文件 {filename} 内容，返回第 {start_index} 行到第 {end_index} 行")
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
            fragment_meta_list: List[DocumentFragmentMeta] = []
            
            # 参数解析
            try:
                if isinstance(input, str):
                    # 解析JSON字符串
                    parsed_input = json.loads(input)
                    
                    # 检查是否是fragments数组格式
                    if "fragments" in parsed_input and isinstance(parsed_input["fragments"], list):
                        # 处理fragments数组格式
                        for item in parsed_input["fragments"]:
                            filename = item.get("filename")
                            start_line = item.get("start_line")
                            end_line = item.get("end_line")
                            if filename and start_line and end_line:
                                fragment_meta_list.append(DocumentFragmentMeta(
                                    filename=filename, 
                                    start_line=start_line, 
                                    end_line=end_line
                                ))
                    else:
                        # 处理单个片段格式
                        filename = parsed_input.get("filename")
                        start_line = parsed_input.get("start_line")
                        end_line = parsed_input.get("end_line")
                        if filename and start_line and end_line:
                            fragment_meta_list.append(DocumentFragmentMeta(
                                filename=filename, 
                                start_line=start_line, 
                                end_line=end_line
                            ))
                elif isinstance(input, AddFragmentMetaToolInput):
                    # 处理Pydantic模型
                    fragment_meta_list.append(DocumentFragmentMeta(
                        filename=input.filename, 
                        start_line=input.start_line, 
                        end_line=input.end_line
                    ))
                elif isinstance(input, list):
                    # 处理列表格式
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
            except Exception as e:
                log.error(f"添加文档片段元数据失败: {e}")
                log.error(f"输入参数: {input}")
                log.error(f"输入类型: {type(input)}")
                return f"输入参数有误，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
            
            # 验证参数
            for fragment in fragment_meta_list:
                if not fragment.filename or not fragment.start_line or not fragment.end_line:
                    return f"输入参数有误，文件名、起始行号和结束行号不能为空，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
                if fragment.start_line > fragment.end_line:
                    return f"输入参数有误，起始行号不能大于结束行号，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
                if fragment.start_line < 1:
                    return f"输入参数有误，起始行号不能小于1，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"
                if fragment.end_line < 1:
                    return f"输入参数有误，结束行号不能小于1，请参考格式: {AddFragmentMetaToolInput.get_example_format()}，重新检查后重试。"

            self.fragments_meta.extend(fragment_meta_list)
            log.info(f"添加文档片段元数据成功: {len(fragment_meta_list)} 个片段")
            return f"添加文档片段元数据成功，共保存 {len(fragment_meta_list)} 个片段"
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
                    "start_index": fragment_meta.start_line,
                    "end_index": fragment_meta.end_line
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
    print(reader_llm.fragments_meta)