#!/usr/bin/env python3
"""
知识库问答系统 - CLI 启动脚本
使用方法: uv run python start_cli.py
"""

import sys
from pathlib import Path

def main():
    print("🚀 启动知识库问答系统 CLI 界面...")
    print("=" * 50)
    
    # 检查是否在正确的目录
    if not Path("pyproject.toml").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        return
    
    # 检查 .env 文件是否存在
    if not Path(".env").exists():
        print("⚠️  警告: 未找到 .env 文件，请确保已配置环境变量")
        print("参考 README.md 中的配置说明")
    
    print("🎯 启动 CLI 界面...")
    print("=" * 50)
    
    # 启动 CLI
    try:
        from src.knowledge_qa.cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请使用: uv run python start_cli.py")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
    
    print("\n👋 CLI 界面已退出")

if __name__ == "__main__":
    main()
