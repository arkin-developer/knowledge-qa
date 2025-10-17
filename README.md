# 知识库问答系统

基于 LangGraph + FAISS + LLM 构建的智能知识库问答系统，支持多种文档格式上传、向量检索和流式对话。

## 🚀 功能特性

- **多格式文档支持**: PDF、Word (.docx)、Markdown (.md)、TXT
- **智能文本分段**: 自动进行合适粒度的文本分段
- **向量化存储**: 使用 FAISS 构建高效的向量索引
- **智能检索**: 基于语义相似度的文档检索
- **流式对话**: 支持实时流式回答生成
- **多轮对话**: 保留上下文，支持连续提问
- **可观测性**: 集成 LangSmith 进行完整的调用链追踪
- **多种界面**: 提供 CLI 和 Web UI 两种使用方式

## 🛠️ 技术栈

- **框架**: LangGraph (工作流编排)
- **向量数据库**: FAISS (向量存储与检索)
- **大语言模型**: Qwen/Qwen3-VL-30B-A3B-Instruct
- **嵌入模型**: Qwen/Qwen3-Embedding-8B
- **Web界面**: Streamlit
- **命令行界面**: Rich
- **日志系统**: Loguru
- **包管理**: UV
- **可观测性**: LangSmith

## 📦 安装与启动

### 环境要求

- Python 3.8+
- UV 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd knowledge-qa
```

2. **安装依赖**
```bash
uv sync
```

3. **配置环境变量**
创建 `.env` 文件并配置以下参数：
```env
# SiliconCloud 配置
SILICONCLOUD_API_KEY=your_api_key
SILICONCLOUD_API_BASE=https://api.siliconflow.cn/v1

# LLM 配置
LLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=256000

# 嵌入模型配置
EMBEDDING_PROVIDER=siliconcloud
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

# 文本分段配置
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# 向量库配置
VECTOR_STORE_PATH=./data/faiss_db
SEARCH_K=3

# 记忆配置
MEMORY_WINDOW_SIZE=30

# LangSmith 配置
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=knowledge_qa_test
LANGCHAIN_TRACING_V2=true
LANGCHAIN_DEBUG=false

# 应用配置
APP_ENV=development
LOG_LEVEL=INFO
```

## 🎯 使用指南

### 1. 命令行界面 (CLI)

**方法一：使用启动脚本（推荐）**
```bash
uv run python start_cli.py
```

**方法二：直接启动**
```bash
uv run python -m src.knowledge_qa.cli
```

CLI 功能菜单：
- `1` - 上传文档到知识库
- `2` - 查看聊天记录上下文
- `3` - 查看目前向量存储的数量
- `4` - 清除上下文
- `5` - 流式问答模式
- `0` - 退出程序
- 直接输入问题 - 开始对话

### 2. Web 界面 (Streamlit)

**方法一：使用启动脚本（推荐）**
```bash
# 使用默认端口 8501
uv run python start_web.py

# 指定端口
uv run python start_web.py 8502
```

**方法二：直接启动**
```bash
uv run streamlit run src/knowledge_qa/app.py --server.port 8501
```

访问 http://localhost:8501 使用 Web 界面。

### 3. 编程接口

```python
from src.knowledge_qa.agent import KnowledgeQAAgent

# 初始化 Agent
agent = KnowledgeQAAgent()

# 上传文档
result = agent.chat("", file_path="path/to/document.pdf")
print(f"上传结果: {result['mode']}")

# 查询知识库
result = agent.chat("您的问题")
print(f"回答: {result['answer']}")
print(f"引用来源: {len(result['sources'])} 个")

# 流式对话
for chunk in agent.chat_streaming("您的问题"):
    if isinstance(chunk, dict):
        print(f"模式: {chunk.get('mode')}")
    else:
        print(chunk, end="", flush=True)
```

## 📁 项目结构

```
knowledge-qa/
├── src/knowledge_qa/          # 核心源代码
│   ├── agent.py              # LangGraph Agent 主逻辑
│   ├── app.py                # Streamlit Web 界面
│   ├── cli.py                # 命令行界面
│   ├── config.py             # 配置管理
│   ├── file_parser.py        # 文档解析器
│   ├── llm.py                # LLM 接口
│   ├── log_manager.py        # 日志管理
│   ├── memory.py             # 对话记忆管理
│   └── text_processor.py     # 文本处理与向量化
├── examples/                 # 示例文档
├── data/                     # 数据存储目录
│   └── faiss_db/            # FAISS 向量数据库
├── logs/                     # 日志文件
├── pyproject.toml           # 项目配置与依赖
├── uv.lock                  # 依赖锁定文件
└── README.md                # 项目说明文档
```

## 🔍 可观测性

系统集成了 LangSmith 进行完整的调用链追踪：

1. 访问 https://smith.langchain.com/
2. 登录您的 LangSmith 账户
3. 查看项目: `knowledge_qa_test`
4. 您将看到完整的调用链追踪信息，包括：
   - 每个节点的执行时间
   - 输入和输出数据
   - 错误信息（如果有）
   - 性能指标和相似度分数

## 🧪 测试

运行完整测试：
```bash
uv run python -m src.knowledge_qa.agent
```

测试 LLM 功能：
```bash
uv run python -m src.knowledge_qa.llm
```

测试文本处理：
```bash
uv run python -m src.knowledge_qa.text_processor
```

## 📊 性能特性

- **向量检索**: 毫秒级响应
- **文档处理**: 支持大文件批量处理
- **流式输出**: 实时响应，提升用户体验
- **内存管理**: 智能的对话上下文管理
- **错误处理**: 完善的异常处理和恢复机制

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如果您遇到任何问题或有建议，请：

1. 查看 [Issues](https://github.com/your-repo/issues) 页面
2. 创建新的 Issue 描述您的问题
3. 联系维护者

---

**注意**: 使用前请确保已正确配置所有必要的 API 密钥和环境变量。
