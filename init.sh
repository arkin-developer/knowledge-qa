#!/bin/bash

# 知识库问答系统初始化脚本
# 使用方法: ./init.sh

set -e

echo "🚀 知识库问答系统初始化开始..."
echo "=================================="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker，请先安装 Docker"
    echo "安装指南: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未找到 Docker Compose，请先安装 Docker Compose"
    echo "安装指南: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 .env 文件是否存在
if [ ! -f ".env" ]; then
    echo "⚠️  警告: 未找到 .env 文件"
    echo "📝 创建示例 .env 文件..."
    
    cat > .env << EOF
# SiliconCloud 配置
SILICONCLOUD_API_KEY=your_api_key_here
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
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=knowledge_qa_test
LANGCHAIN_TRACING_V2=true
LANGCHAIN_DEBUG=false

# 应用配置
APP_ENV=development
LOG_LEVEL=INFO
EOF
    
    echo "✅ 已创建 .env 文件，请编辑其中的配置"
    echo "⚠️  请务必配置正确的 API 密钥后再继续"
    read -p "按 Enter 继续..."
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data/faiss_db logs

# 构建 Docker 镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

echo ""
echo "✅ 初始化完成！"
echo "=================================="
echo "🌐 Web 界面: http://localhost:8501"
echo "📊 服务状态: docker-compose ps"
echo "📝 查看日志: docker-compose logs -f"
echo "🛑 停止服务: docker-compose down"
echo "=================================="
