"""命令行界面模块"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .agent import KnowledgeQAAgent
from .log_manager import log
from .config import settings


class CLI:
    """命令行界面类"""

    def __init__(self):
        self.console = Console()
        self.agent = KnowledgeQAAgent()
        self.temp_upload_dir = Path(settings.upload_temp_path)
        self._ensure_temp_dir()

    def _ensure_temp_dir(self):
        """确保临时上传目录存在"""
        if not self.temp_upload_dir.exists():
            self.temp_upload_dir.mkdir(parents=True, exist_ok=True)
            self.console.print(f"[green]✅ 创建临时上传目录: {self.temp_upload_dir}[/green]")

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
            "6. 管理临时上传文件夹",
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
        
        # 显示临时上传文件夹中的文件
        temp_files = []
        if self.temp_upload_dir.exists():
            temp_files = [f for f in self.temp_upload_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.md', '.txt']]
        
        # 显示 examples 文件夹中的测试文件
        examples_dir = Path("examples")
        examples_files = []
        if examples_dir.exists():
            examples_files = [f for f in examples_dir.iterdir() 
                            if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.md', '.txt']]
        
        all_files = temp_files + examples_files
        file_offset = 0
        
        if temp_files:
            self.console.print("[bold cyan]📂 临时上传文件夹中的文件:[/bold cyan]")
            for i, file_path in enumerate(temp_files, 1):
                self.console.print(f"  [cyan]{i}.[/cyan] {file_path}")
            file_offset = len(temp_files)
            self.console.print()
        
        if examples_files:
            self.console.print("[bold cyan]📂 示例文件夹中的文件:[/bold cyan]")
            for i, file_path in enumerate(examples_files, 1):
                self.console.print(f"  [cyan]{i + file_offset}.[/cyan] {file_path}")
            self.console.print()
        
        if all_files:
            self.console.print("[dim]💡 提示: 输入数字选择文件，或输入完整路径选择其他文件[/dim]")
            self.console.print()
        
        user_input = Prompt.ask("请输入文件路径或数字")
        
        if not user_input:
            return
        
        # 处理数字选择
        if user_input.isdigit():
            file_index = int(user_input) - 1
            if 0 <= file_index < len(all_files):
                file_path = all_files[file_index]
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
                
                # 如果文件不在临时文件夹中，复制到临时文件夹保存
                if not str(file_path).startswith(str(self.temp_upload_dir)):
                    temp_file_path = self.temp_upload_dir / file_path.name
                    shutil.copy2(file_path, temp_file_path)
                    self.console.print(f"[green]📁 文件已保存到临时文件夹: {temp_file_path}[/green]")
                else:
                    self.console.print(f"[green]📁 文件保留在临时文件夹: {file_path}[/green]")
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
            # 获取向量库信息
            vector_store = self.agent.vector_store

            table = Table(show_header=False, box=None)
            table.add_column("属性", style="cyan")
            table.add_column("值", style="green")

            if vector_store is None or vector_store._vector_store is None:
                table.add_row("状态", "未初始化")
                table.add_row("文档数量", "0")
            else:
                table.add_row("状态", "已初始化")
                try:
                    # 使用VectorStore的get_vector_store_info方法获取信息
                    store_info = vector_store.get_vector_store_info()
                    table.add_row("文档数量", str(store_info.get("document_count", "未知")))
                    table.add_row("存储路径", str(store_info.get("persist_path", "未知")))
                except Exception as e:
                    table.add_row("文档数量", "无法获取")
                    table.add_row("错误信息", str(e))

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
                document_fragments = []
                vector_docs = []

                for chunk in self.agent.chat_streaming(query):
                    if isinstance(chunk, dict):
                        # 最后的元数据
                        document_fragments = chunk.get("document_fragments", [])
                        vector_docs = chunk.get("vector_docs", [])
                        mode = chunk.get("mode", "unknown")
                    else:
                        # 流式文本内容
                        self.console.print(chunk, end="")

                self.console.print()  # 换行

                # 优先显示 document_fragments，如果没有则显示 vector_docs
                if document_fragments and len(document_fragments) > 0:
                    sources = []
                    for i, fragment in enumerate(document_fragments, 1):
                        if hasattr(fragment, 'content'):
                            content = fragment.content[:100] + "..." if len(fragment.content) > 100 else fragment.content
                            sources.append({
                                "index": i,
                                "content": content,
                                "filename": fragment.filename,
                                "lines": f"{fragment.start_line}-{fragment.end_line}"
                            })
                        else:
                            # 处理字典格式
                            content = fragment.get('content', '')[:100] + "..." if len(fragment.get('content', '')) > 100 else fragment.get('content', '')
                            sources.append({
                                "index": i,
                                "content": content,
                                "filename": fragment.get('filename', ''),
                                "lines": f"{fragment.get('start_line', '')}-{fragment.get('end_line', '')}"
                            })
                elif vector_docs and len(vector_docs) > 0:
                    sources = []
                    for i, doc in enumerate(vector_docs, 1):
                        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        sources.append({
                            "index": i,
                            "content": content,
                            "filename": doc.metadata.get('filename', '未知'),
                            "lines": f"{doc.metadata.get('start_line', '')}-{doc.metadata.get('end_line', '')}" if doc.metadata.get('start_line') else "未知"
                        })

                if sources:
                    self.console.print(f"\n[bold cyan]📚 引用来源 (共{len(sources)}条):[/bold cyan]")
                    for source in sources:
                        self.console.print(
                            f"  [cyan][{source['index']}][/cyan] {source['content']}")
                        if source.get('filename'):
                            self.console.print(f"    [dim]文件: {source['filename']} 行号: {source['lines']}[/dim]")

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
            answer = result.get("qa_answer", "无法生成回答")
            self.console.print(Markdown(answer))

            # 优先显示 document_fragments，如果没有则显示 vector_docs
            document_fragments = result.get("document_fragments", [])
            vector_docs = result.get("vector_docs", [])
            
            sources = []
            if document_fragments and len(document_fragments) > 0:
                # 显示 document_fragments
                sources = []
                for i, fragment in enumerate(document_fragments, 1):
                    if hasattr(fragment, 'content'):
                        content = fragment.content[:100] + "..." if len(fragment.content) > 100 else fragment.content
                        sources.append({
                            "index": i,
                            "content": content,
                            "filename": fragment.filename,
                            "lines": f"{fragment.start_line}-{fragment.end_line}"
                        })
                    else:
                        # 处理字典格式
                        content = fragment.get('content', '')[:100] + "..." if len(fragment.get('content', '')) > 100 else fragment.get('content', '')
                        sources.append({
                            "index": i,
                            "content": content,
                            "filename": fragment.get('filename', ''),
                            "lines": f"{fragment.get('start_line', '')}-{fragment.get('end_line', '')}"
                        })
            elif vector_docs and len(vector_docs) > 0:
                # 显示 vector_docs
                sources = []
                for i, doc in enumerate(vector_docs, 1):
                    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    sources.append({
                        "index": i,
                        "content": content,
                        "filename": doc.metadata.get('filename', '未知'),
                        "lines": f"{doc.metadata.get('start_line', '')}-{doc.metadata.get('end_line', '')}" if doc.metadata.get('start_line') else "未知"
                    })
            
            if sources:
                self.console.print(f"\n[bold cyan]📚 引用来源 (共{len(sources)}条):[/bold cyan]")
                for source in sources:
                    self.console.print(
                        f"  [cyan][{source['index']}][/cyan] {source['content']}")
                    if source.get('filename'):
                        self.console.print(f"    [dim]文件: {source['filename']} 行号: {source['lines']}[/dim]")

            self.console.print(
                f"\n[dim]模式: {result.get('mode', 'unknown')}[/dim]")

        except Exception as e:
            self.console.print(f"[red]❌ 错误: {str(e)}[/red]")
            log.error(f"查询失败: {str(e)}")

    def handle_manage_temp_folder(self):
        """管理临时上传文件夹"""
        self.console.print("\n[bold blue]📁 临时上传文件夹管理[/bold blue]")
        self.console.print()
        
        if not self.temp_upload_dir.exists():
            self.console.print("[yellow]临时文件夹不存在[/yellow]")
            return
        
        # 显示临时文件夹中的文件
        temp_files = [f for f in self.temp_upload_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.md', '.txt']]
        
        if not temp_files:
            self.console.print("[yellow]临时文件夹为空[/yellow]")
            return
        
        self.console.print(f"[green]临时文件夹路径: {self.temp_upload_dir}[/green]")
        self.console.print(f"[green]文件数量: {len(temp_files)}[/green]")
        self.console.print()
        
        # 显示文件列表
        for i, file_path in enumerate(temp_files, 1):
            file_size = file_path.stat().st_size
            size_str = f"{file_size / 1024:.1f}KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f}MB"
            self.console.print(f"  [cyan]{i}.[/cyan] {file_path.name} ({size_str})")
        
        self.console.print()
        self.console.print("[dim]💡 提示: 文件已永久保存在临时文件夹中，不会自动删除[/dim]")

    def run(self):
        """运行CLI"""
        try:
            while True:
                self.display_welcome()
                self.display_menu()

                user_input = Prompt.ask("请选择或直接输入问题")

                if not user_input:
                    # 如果直接按回车，显示提示并继续循环
                    self.console.print("[yellow]💡 请输入选项数字(0-5)或直接输入问题，按Ctrl+C退出[/yellow]")
                    continue
                elif user_input == "0":
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
                elif user_input == "6":
                    self.handle_manage_temp_folder()
                elif user_input in ["exit", "quit", "q"]:
                    self.console.print("\n[green]👋 再见！[/green]")
                    break
                else:
                    # 检查是否为纯数字
                    if user_input.isdigit():
                        self.console.print("[red]❌ 无效选项，请输入 0-6 之间的数字[/red]")
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


def main():
    """主函数"""
    cli = CLI()
    cli.run()

if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.cli
    main()
