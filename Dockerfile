# 知识库问答系统 Dockerfile
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 使用清华大学镜像源加速（中国大陆用户）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖（添加重试机制）
RUN apt-get update && apt-get install -y --fix-missing \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（使用 pip 安装，最简单可靠）
RUN pip install --no-cache-dir uv

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY start_cli.py start_web.py ./

# 安装 Python 依赖（让 uv 灵活处理依赖版本）
RUN uv sync --python $(which python3)

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
