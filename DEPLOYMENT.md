# 部署指南

本文档提供了知识库问答系统的完整部署指南，包括 Docker 部署、环境配置和运维管理。

## 🐳 Docker 部署

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/arkin-developer/knowledge-qa.git
cd knowledge-qa
```

2. **一键初始化部署**
```bash
chmod +x init.sh
./init.sh
```

3. **访问应用**
- Web 界面: http://localhost:8501
- 健康检查: http://localhost:8501/_stcore/health

### 手动部署

#### 1. 环境配置

创建 `.env` 文件：
```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的 API 密钥
```

#### 2. Docker 构建

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 3. 停止服务

```bash
# 停止服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

## 🔧 环境配置

### 必需的环境变量

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| `SILICONCLOUD_API_KEY` | SiliconCloud API 密钥 | `sk-xxx...` |
| `SILICONCLOUD_API_BASE` | API 基础 URL | `https://api.siliconflow.cn/v1` |
| `LLM_MODEL` | LLM 模型名称 | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `Qwen/Qwen3-Embedding-8B` |
| `LANGSMITH_API_KEY` | LangSmith API 密钥 | `lsv2_pt_xxx...` |
| `LANGSMITH_PROJECT` | LangSmith 项目名 | `knowledge_qa_test` |

### 可选的环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `LLM_TEMPERATURE` | LLM 温度参数 | `0.7` |
| `CHUNK_SIZE` | 文本分块大小 | `600` |
| `SEARCH_K` | 检索文档数量 | `3` |
| `MEMORY_WINDOW_SIZE` | 记忆窗口大小 | `30` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 🏗️ 服务架构

### 组件说明

```
┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │  Knowledge-QA   │
│  (反向代理)      │────│   (主应用)      │
│   Port: 80/443  │    │   Port: 8501    │
└─────────────────┘    └─────────────────┘
```

### 数据持久化

- **向量数据库**: `./data/faiss_db/` - FAISS 向量索引文件
- **日志文件**: `./logs/` - 应用日志

## 🚀 生产环境部署

### 1. 使用 Nginx 反向代理

```bash
# 启动包含 Nginx 的完整服务
docker-compose --profile production up -d
```

### 2. SSL 证书配置

1. 将 SSL 证书文件放置到 `./ssl/` 目录
2. 修改 `nginx.conf` 中的域名配置
3. 取消注释 HTTPS 配置部分

### 3. 环境变量管理

生产环境建议使用 Docker Secrets 或外部配置管理：

```bash
# 使用 Docker Secrets
echo "your-api-key" | docker secret create siliconcloud_api_key -
```

## 📊 监控与运维

### 健康检查

```bash
# 检查服务健康状态
curl http://localhost:8501/_stcore/health

# 检查容器状态
docker-compose ps

# 查看资源使用情况
docker stats
```

### 日志管理

```bash
# 查看实时日志
docker-compose logs -f knowledge-qa

# 查看特定时间段的日志
docker-compose logs --since="2024-01-01T00:00:00" knowledge-qa

# 导出日志
docker-compose logs knowledge-qa > app.log
```

### 数据备份

```bash
# 备份向量数据库
tar -czf faiss_backup_$(date +%Y%m%d).tar.gz data/faiss_db/
```

### 性能优化

1. **资源限制**
```yaml
# 在 docker-compose.yml 中添加
services:
  knowledge-qa:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 🔄 CI/CD 集成

### GitHub Actions

项目已配置 GitHub Actions 工作流：

- **测试**: 代码检查、类型检查、单元测试
- **构建**: Docker 镜像构建和推送
- **部署**: 自动部署到生产环境

### 手动触发部署

```bash
# 构建并推送镜像
docker build -t arkin-developer/knowledge-qa:latest .
docker push arkin-developer/knowledge-qa:latest

# 更新生产环境
docker-compose pull
docker-compose up -d
```

## 🛠️ 故障排除

### 常见问题

1. **端口冲突**
```bash
# 检查端口占用
netstat -tulpn | grep :8501

# 修改端口
docker-compose up -d -p 8502:8501
```

2. **内存不足**
```bash
# 检查内存使用
docker stats

# 增加内存限制
docker-compose up -d --scale knowledge-qa=1
```

3. **API 密钥错误**
```bash
# 检查环境变量
docker-compose exec knowledge-qa env | grep API_KEY

# 重新配置
docker-compose down
# 编辑 .env 文件
docker-compose up -d
```

### 调试模式

```bash
# 启用调试模式
echo "LANGCHAIN_DEBUG=true" >> .env
docker-compose up -d

# 查看详细日志
docker-compose logs -f knowledge-qa
```

## 📞 支持

如果遇到部署问题，请：

1. 查看 [Issues](https://github.com/arkin-developer/knowledge-qa/issues)
2. 检查日志文件
3. 提交新的 Issue 描述问题

---

**注意**: 生产环境部署前请确保已正确配置所有必要的 API 密钥和安全设置。
