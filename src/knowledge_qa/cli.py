"""命令行界面模块"""

import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .agent import KnowledgeQAAgent
from .log_manager import log


class CLI:
    """命令行界面类"""

    def __init__(self):
        self.console = Console()
        self.agent = KnowledgeQAAgent()

    def display_welcome(self):
        """显示欢迎信息"""
        self.console.clear()
        self.console.print(
            Panel(
                "[bold cyan]知识库问答系统[/bold cyan]\n"
                "[green]基于 FAISS + LangGraph[/green]",
                title="欢迎",
                expand=False
            )
        )
        self.console.print()

    def display_menu(self):
        """显示主菜单"""
        menu = [
            "1. 上传文档到知识库",
            "2. 查看聊天记录上下文",
            "3. 查看目前向量存储的数量",
            "4. 清除上下文",
            "5. 流式问答模式",
            "0. 退出"
        ]

        for item in menu:
            self.console.print(f"  [yellow]{item}[/yellow]")
        self.console.print()
        self.console.print("  [dim]💡 提示: 直接输入问题即可开始对话[/dim]")
        self.console.print()

    def handle_upload_document(self):
        """处理文档上传"""
        self.console.print("\n[bold blue]📁 文档上传[/bold blue]")
        self.console.print("支持格式: PDF, DOCX, Markdown, TXT")
        self.console.print()
        
        # 显示 examples 文件夹中的测试文件
        examples_dir = Path("examples")
        if examples_dir.exists():
            self.console.print("[bold cyan]📂 可用的测试文件:[/bold cyan]")
            test_files = []
            for i, file_path in enumerate(examples_dir.iterdir(), 1):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.md', '.txt']:
                    test_files.append(file_path)
                    self.console.print(f"  [cyan]{i}.[/cyan] {file_path}")
            
            if test_files:
                self.console.print()
                self.console.print("[dim]💡 提示: 输入数字选择测试文件，或输入完整路径选择其他文件[/dim]")
                self.console.print()
        
        user_input = Prompt.ask("请输入文件路径或数字")
        
        if not user_input:
            return
        
        # 处理数字选择
        if user_input.isdigit():
            file_index = int(user_input) - 1
            if 0 <= file_index < len(test_files):
                file_path = test_files[file_index]
                self.console.print(f"[green]已选择: {file_path}[/green]")
            else:
                self.console.print("[red]❌ 无效的数字选择[/red]")
                return
        else:
            # 处理路径输入
            file_path = Path(user_input).expanduser().resolve()
            
            if not file_path.exists():
                self.console.print("[red]❌ 文件不存在[/red]")
                return

        try:
            with self.console.status("正在处理文档...", spinner="dots"):
                result = self.agent.chat("", file_path=str(file_path))

            if result.get("mode") == "upload":
                self.console.print(
                    f"[green]✅ 文档上传成功！模式: {result['mode']}[/green]")
            else:
                self.console.print(
                    f"[yellow]⚠️ 处理完成，模式: {result.get('mode', 'unknown')}[/yellow]")

        except Exception as e:
            self.console.print(f"[red]❌ 错误: {str(e)}[/red]")
            log.error(f"上传文档失败: {str(e)}")

    def handle_view_chat_history(self):
        """查看聊天记录上下文"""
        self.console.print("\n[bold blue]📜 聊天记录上下文[/bold blue]")
        self.console.print()

        try:
            messages = self.agent.llm.memory.get_messages()

            if not messages:
                self.console.print("[yellow]当前没有聊天记录[/yellow]")
                return

            self.console.print(f"[green]当前共有 {len(messages)} 条聊天记录[/green]")
            self.console.print()

            for i, msg in enumerate(messages[-10:], 1):  # 显示最近10条
                role = "用户" if msg.type == "human" else "AI助手"
                content = msg.content[:100] + \
                    "..." if len(msg.content) > 100 else msg.content
                self.console.print(f"[cyan]{i}.[/cyan] [{role}] {content}")

            if len(messages) > 10:
                self.console.print(
                    f"[dim]... 还有 {len(messages) - 10} 条记录[/dim]")

        except Exception as e:
            self.console.print(f"[red]❌ 获取聊天记录失败: {str(e)}[/red]")

    def handle_view_vector_store_info(self):
        """查看向量库信息"""
        self.console.print("\n[bold blue]📊 向量库信息[/bold blue]")
        self.console.print()

        try:
            # 尝试获取向量库信息
            vector_store = self.agent.text_processor.vector_store

            table = Table(show_header=False, box=None)
            table.add_column("属性", style="cyan")
            table.add_column("值", style="green")

            if vector_store is None:
                table.add_row("状态", "未初始化")
                table.add_row("文档数量", "0")
            else:
                table.add_row("状态", "已初始化")
                try:
                    # 尝试获取文档数量
                    doc_count = len(vector_store.docstore._dict) if hasattr(
                        vector_store, 'docstore') else "未知"
                    table.add_row("文档数量", str(doc_count))
                except:
                    table.add_row("文档数量", "无法获取")

                # 获取存储路径
                persist_path = getattr(vector_store, 'persist_path', None)
                if persist_path:
                    table.add_row("存储路径", str(persist_path))

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]❌ 获取信息失败: {str(e)}[/red]")

    def handle_clear_context(self):
        """清除上下文"""
        if Confirm.ask("确定要清空所有聊天记录吗？"):
            try:
                self.agent.clear_memory()
                self.console.print("[green]✅ 聊天记录已清空[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ 清空失败: {str(e)}[/red]")

    def handle_streaming_query(self):
        """处理流式查询"""
        self.console.print("\n[bold blue]💬 流式问答模式[/bold blue]")
        self.console.print("输入 'exit' 返回主菜单，输入 'clear' 清空屏幕")
        self.console.print()

        while True:
            try:
                query = Prompt.ask("\n[bold green]你[/bold green]")

                if not query:
                    continue

                if query.lower() == "exit":
                    break
                elif query.lower() == "clear":
                    self.console.clear()
                    continue

                self.console.print("\n[bold blue]🤖 AI助手[/bold blue]")
                self.console.print("", end="")

                sources = []
                mode = None

                for chunk in self.agent.chat_streaming(query):
                    if isinstance(chunk, dict):
                        # 最后的元数据
                        sources = chunk.get("sources", [])
                        mode = chunk.get("mode", "unknown")
                    else:
                        # 流式文本内容
                        self.console.print(chunk, end="")

                self.console.print()  # 换行

                if sources:
                    self.console.print("\n[bold cyan]📚 引用来源:[/bold cyan]")
                    for source in sources[:3]:
                        content = source.get("content", "")[:100]
                        self.console.print(
                            f"  [cyan][{source.get('index')}][/cyan] {content}...")

                self.console.print(f"\n[dim]模式: {mode}[/dim]")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]已中断[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ 错误: {str(e)}[/red]")
                log.error(f"流式查询失败: {str(e)}")

    def handle_direct_query(self, query: str):
        """处理直接输入的问题"""
        try:
            with self.console.status("正在思考...", spinner="dots"):
                result = self.agent.chat(query)

            self.console.print("\n[bold blue]🤖 AI助手[/bold blue]")
            answer = result.get("answer", "无法生成回答")
            self.console.print(Markdown(answer))

            sources = result.get("sources", [])
            if sources:
                self.console.print("\n[bold cyan]📚 引用来源:[/bold cyan]")
                for source in sources[:3]:
                    content = source.get("content", "")[:100]
                    self.console.print(
                        f"  [cyan][{source.get('index')}][/cyan] {content}...")

            self.console.print(
                f"\n[dim]模式: {result.get('mode', 'unknown')}[/dim]")

        except Exception as e:
            self.console.print(f"[red]❌ 错误: {str(e)}[/red]")
            log.error(f"查询失败: {str(e)}")

    def run(self):
        """运行CLI"""
        try:
            while True:
                self.display_welcome()
                self.display_menu()

                user_input = Prompt.ask("请选择或直接输入问题", default="0")

                if not user_input or user_input == "0":
                    self.console.print("\n[green]👋 再见！[/green]")
                    break
                elif user_input == "1":
                    self.handle_upload_document()
                elif user_input == "2":
                    self.handle_view_chat_history()
                elif user_input == "3":
                    self.handle_view_vector_store_info()
                elif user_input == "4":
                    self.handle_clear_context()
                elif user_input == "5":
                    self.handle_streaming_query()
                elif user_input in ["exit", "quit", "q"]:
                    self.console.print("\n[green]👋 再见！[/green]")
                    break
                else:
                    # 检查是否为纯数字
                    if user_input.isdigit():
                        self.console.print("[red]❌ 无效选项，请输入 0-5 之间的数字[/red]")
                    else:
                        self.console.print(f"\n[dim]你的问题: {user_input}[/dim]")
                        self.handle_direct_query(user_input)

                self.console.print()
                Prompt.ask("按Enter继续")

        except KeyboardInterrupt:
            self.console.print("\n[green]👋 再见！[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ 程序错误: {str(e)}[/red]")
            log.error(f"CLI错误: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.cli
    cli = CLI()
    cli.run()
