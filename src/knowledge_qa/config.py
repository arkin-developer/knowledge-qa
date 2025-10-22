"""全局配置"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # SiliconCloud 配置
    siliconcloud_api_key: str
    siliconcloud_api_base: str

    # LLM配置
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Embedding 配置
    embedding_provider: str
    embedding_model: str

    # 文本分段配置
    chunk_size: int
    chunk_overlap: int

    # 向量库配置
    vector_store_path: str

    # 文件上传配置
    upload_temp_path: str

    # 搜索配置
    search_k: int

    # 记忆配置
    memory_window_size: int

    # LangSmith 配置
    langsmith_api_key: str
    langsmith_project: str
    langchain_tracing_v2: str
    langchain_debug: str

    # 应用配置
    app_env: str
    log_level: str

    class Config:
        """在.env文件中配置"""
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

# 自动设置 LangSmith 环境变量
import os
os.environ['LANGCHAIN_TRACING_V2'] = settings.langchain_tracing_v2
os.environ['LANGCHAIN_API_KEY'] = settings.langsmith_api_key
os.environ['LANGCHAIN_PROJECT'] = settings.langsmith_project

if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.config
    print(settings)
