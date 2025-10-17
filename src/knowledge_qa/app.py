"""Streamlit Web 应用入口"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

try:
    # 尝试相对导入（当作为模块运行时）
    from .agent import KnowledgeQAAgent
    from .log_manager import log
except ImportError:
    # 使用绝对导入（当直接运行时）
    import sys
    from pathlib import Path

    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.knowledge_qa.agent import KnowledgeQAAgent
    from src.knowledge_qa.log_manager import log


class StreamlitApp:
    """Streamlit Web 应用类"""

    def __init__(self):
        self.agent: Optional[KnowledgeQAAgent] = None
        self._initialize_session_state()

    def _initialize_session_state(self):
        """初始化会话状态"""
        if "agent" not in st.session_state:
            st.session_state.agent = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "vector_store_info" not in st.session_state:
            st.session_state.vector_store_info = None

    def _get_agent(self) -> KnowledgeQAAgent:
        """获取或创建 Agent 实例"""
        if st.session_state.agent is None:
            with st.spinner("正在初始化知识库问答系统..."):
                st.session_state.agent = KnowledgeQAAgent()
        return st.session_state.agent

    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("📚 知识库管理")

            # 文件上传区域
            st.subheader("📁 文档上传")
            uploaded_file = st.file_uploader(
                "选择文档文件",
                type=['pdf', 'docx', 'md', 'txt'],
                help="支持 PDF、DOCX、Markdown、TXT 格式"
            )

            if uploaded_file is not None:
                if st.button("上传到知识库", type="primary"):
                    self._handle_file_upload(uploaded_file)

            # 显示已上传的文件
            if st.session_state.uploaded_files:
                st.subheader("📋 已上传文件")
                for file_info in st.session_state.uploaded_files:
                    st.text(f"• {file_info['name']}")

            st.divider()

            # 向量库信息
            st.subheader("📊 向量库状态")
            self._display_vector_store_info()

            st.divider()

            # 操作按钮
            st.subheader("🔧 操作")
            if st.button("清空聊天记录", type="secondary"):
                self._clear_chat_history()

            if st.button("刷新向量库信息", type="secondary"):
                self._refresh_vector_store_info()

    def _handle_file_upload(self, uploaded_file):
        """处理文件上传"""
        try:
            # 保存上传的文件到临时位置
            temp_path = Path("temp") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 使用 Agent 处理文件
            agent = self._get_agent()
            with st.spinner(f"正在处理文件: {uploaded_file.name}"):
                result = agent.chat("", file_path=str(temp_path))

            if result.get("mode") == "upload":
                st.success(f"✅ 文件 {uploaded_file.name} 上传成功！")
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "type": uploaded_file.type
                })
            else:
                st.warning(f"⚠️ 文件处理完成，但模式为: {result.get('mode', 'unknown')}")

            # 清理临时文件
            temp_path.unlink()

        except Exception as e:
            st.error(f"❌ 文件上传失败: {str(e)}")
            log.error(f"文件上传失败: {str(e)}")

    def _display_vector_store_info(self):
        """显示向量库信息"""
        try:
            agent = self._get_agent()
            vector_store = agent.text_processor.vector_store

            if vector_store is None:
                st.info("📭 向量库未初始化")
                st.text("请先上传文档")
            else:
                st.success("✅ 向量库已初始化")
                try:
                    doc_count = len(vector_store.docstore._dict) if hasattr(
                        vector_store, 'docstore') else "未知"
                    st.metric("文档数量", doc_count)
                except:
                    st.text("文档数量: 无法获取")

                persist_path = getattr(vector_store, 'persist_path', None)
                if persist_path:
                    st.text(f"存储路径: {persist_path}")

        except Exception as e:
            st.error(f"获取向量库信息失败: {str(e)}")

    def _clear_chat_history(self):
        """清空聊天记录"""
        st.session_state.messages = []
        try:
            agent = self._get_agent()
            agent.clear_memory()
            st.success("✅ 聊天记录已清空")
        except Exception as e:
            st.error(f"清空失败: {str(e)}")

    def _refresh_vector_store_info(self):
        """刷新向量库信息"""
        st.rerun()

    def render_main_interface(self):
        """渲染主界面"""
        st.title("🤖 知识库问答系统")
        st.markdown("基于 FAISS + LangGraph 构建的智能问答系统")
        
        # 系统信息固定在顶部
        self.render_additional_info()
        
        # 聊天界面
        self._render_chat_interface()

    def _render_chat_interface(self):
        """渲染聊天界面"""
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 显示引用来源
                if message.get("sources"):
                    with st.expander("📚 引用来源"):
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.text(
                                f"[{i}] {source.get('content', '')[:100]}...")

        # 聊天输入
        if prompt := st.chat_input("请输入您的问题..."):
            # 添加用户消息
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 使用流式输出生成 AI 回答
            with st.chat_message("assistant"):
                try:
                    agent = self._get_agent()

                    # 创建占位符
                    message_placeholder = st.empty()
                    sources_placeholder = st.empty()
                    status_placeholder = st.empty()

                    full_response = ""
                    sources = []
                    mode = None
                    processing_stage = "初始化"

                    # 流式生成回答
                    for chunk in agent.chat_streaming(prompt):
                        if isinstance(chunk, dict):
                            # 检查是否是状态消息
                            if chunk.get("type") == "status":
                                status_placeholder.info(f"🔄 {chunk.get('status', '处理中...')}")
                            else:
                                # 最后的元数据
                                sources = chunk.get("sources", [])
                                mode = chunk.get("mode", "unknown")
                                # 清除状态提示
                                status_placeholder.empty()
                        else:
                            # 流式文本内容
                            if not full_response:  # 第一次收到文本内容
                                status_placeholder.info("🤖 正在生成回答...")
                            
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    # 最终显示
                    message_placeholder.markdown(full_response)

                    # 显示引用来源
                    if sources:
                        with sources_placeholder.expander("📚 引用来源"):
                            for i, source in enumerate(sources[:3], 1):
                                st.text(
                                    f"[{i}] {source.get('content', '')[:100]}...")

                    # 显示模式信息
                    st.caption(f"模式: {mode}")

                    # 保存到聊天历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "mode": mode
                    })

                except Exception as e:
                    error_msg = f"❌ 处理失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    def render_additional_info(self):
        """渲染额外信息"""
        # 使用容器固定系统信息位置
        with st.container():
            with st.expander("ℹ️ 系统信息", expanded=False):
                st.markdown("""
                **系统特性:**
                - 🤖 基于 FAISS + LangGraph 构建
                - 💬 实时流式回答
                - 📚 智能文档检索
                - 🔄 上下文记忆管理
                - 📊 向量化知识存储
                
                **支持格式:**
                - PDF 文档
                - Word 文档 (.docx)
                - Markdown 文件 (.md)
                - 纯文本文件 (.txt)
                """)

    def run(self):
        """运行 Streamlit 应用"""
        # 页面配置
        st.set_page_config(
            page_title="知识库问答系统",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 渲染界面
        self.render_sidebar()
        self.render_main_interface()


def main():
    """主函数"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run streamlit run src/knowledge_qa/app.py --server.port 8501 --server.address localhost
    main()
        