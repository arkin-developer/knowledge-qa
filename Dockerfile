# 知识库问答系统 Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY start_cli.py start_web.py ./

# 安装 Python 依赖
RUN uv sync --frozen

# 创建必要的目录
RUN mkdir -p data/faiss_db logs

# 设置权限
RUN chmod +x start_cli.py start_web.py

# 暴露端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 默认启动 Web 界面
CMD ["uv", "run", "python", "start_web.py", "8501"]
