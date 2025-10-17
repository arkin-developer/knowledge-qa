#!/usr/bin/env python3
"""
知识库问答系统 - Web 界面启动脚本
使用方法: uv run python start_web.py [端口号]
默认端口: 8501
"""

import sys
import subprocess
from pathlib import Path

def main():
    # 获取端口参数
    port = sys.argv[1] if len(sys.argv) > 1 else "8501"
    
    print("🌐 启动知识库问答系统 Web 界面...")
    print("=" * 50)
    
    # 检查是否在正确的目录
    if not Path("pyproject.toml").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        return
    
    # 检查 .env 文件是否存在
    if not Path(".env").exists():
        print("⚠️  警告: 未找到 .env 文件，请确保已配置环境变量")
        print("参考 README.md 中的配置说明")
    
    print(f"🎯 启动 Web 界面...")
    print(f"端口: {port}")
    print(f"访问地址: http://localhost:{port}")
    print("=" * 50)
    
    # 启动 Streamlit
    try:
        cmd = [
            "streamlit", "run", "src/knowledge_qa/app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except FileNotFoundError:
        print("❌ 错误: 未找到 streamlit 命令")
        print("请使用: uv run python start_web.py")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
    
    print("\n👋 Web 界面已退出")

if __name__ == "__main__":
    main()
