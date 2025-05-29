import json
import os
import re
import sys
import time
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

# 设置系统编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


DF_AGENT_PROMPT_TEMPLATE = """
你是一个数据分析专家，请根据以下数据框和用户问题提供回答。
必须严格按照指定格式回复，否则系统将无法解析！

**数据框预览**:
{df_head}

**用户问题**: {query}

**响应格式要求**:
- 纯文本回答 (如果没有可视化需求)
- 或JSON格式 (如果需要展示图表):
{{
    "answer": "详细的文本解释和统计结果:",
    "charts": [
        {{
            "type": "bar/line/pie/scatter/box/hist/area",
            "data": {{
                "columns": ["类别A", "类别B", ...],
                "data": [数值1, 数值2, ...]
            }},
            "title": "图表标题 (可选)"
        }}
    ]
}}

**重要规则**:
1. 处理日期时使用 'ME' 代替 'M'
2. 如果遇到错误，返回 JSON 格式的错误信息: {{"error": "错误描述", "answer": "错误信息"}}
3. 不要包含任何额外解释或代码
4. 用户没有明确要求图表时，请仅返回文本答案
5. 不要返回数据预览内容，用户已经在上传文件时看到数据预览
6. 对于需要代码计算的问题，使用以下格式调用工具:
   Thought: 分析问题并确定需要使用的工具
   Action: python_repl_ast
   Action Input: 要执行的 Python 代码（确保代码简洁完整）
7. 日期处理请使用: pd.to_datetime(df['列名'], format='%Y-%m-%d')
8. 不要尝试直接执行代码，必须通过工具调用
9. 生成图表时，不要尝试保存文件，直接返回图表数据
10. 确保月份格式统一为 "YYYY-MM"（例如 "2020-03"）
11. 确保每个月份数据唯一，没有重复
12. 图表数据格式必须严格遵循:
    {{
        "type": "line/bar/pie/scatter/box/hist/area",
        "data": {{
            "columns": ["月份", "销售额"],
            "data": [
                ["2020-01", 5409855],
                ["2020-02", 4608455],
                ...
            ]
        }},
        "title": "图表标题 (可选)"
    }}
13. 确保图表数据中每个数组元素都包含 2 个值：[月份, 销售额]
14. 月份格式必须统一为 "YYYY-MM"（例如 "2020-03"）
15. 确保每个月份数据唯一，没有重复
16. 确保JSON格式正确：键名使用双引号，字符串值使用双引号，数值不使用引号
"""

# 文本代理提示词模板
TEXT_AGENT_PROMPT_TEMPLATE = "你是一个乐于助人的AI助手，用中文回答问题"

# RAG代理提示词模板
RAG_AGENT_PROMPT_TEMPLATE = "你是一个专业的数据分析助手，请根据用户的数据或问题提供准确、详细的回答"


def process_uploaded_file(uploaded_file):
    """处理上传的文件"""
    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        session_id = str(uuid.uuid4())

        if file_ext in ["csv", "xlsx"]:
            if file_ext == "csv":
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.current_mode = "📊 数据分析"
                st.success("CSV文件已成功加载！")
            else:  # xlsx
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names

                if 'selected_sheet' not in st.session_state:
                    st.session_state.selected_sheet = sheet_names[0] if sheet_names else None

                if st.session_state.selected_sheet not in sheet_names:
                    st.session_state.selected_sheet = sheet_names[0] if sheet_names else None

                with st.sidebar:
                    st.subheader("Excel工作表选择")
                    if sheet_names:
                        current_index = sheet_names.index(
                            st.session_state.selected_sheet) if st.session_state.selected_sheet in sheet_names else 0

                        selected_sheet = st.selectbox(
                            "请选择要分析的工作表",
                            sheet_names,
                            index=current_index
                        )

                        st.session_state.selected_sheet = selected_sheet
                        st.session_state.df = excel_file.parse(selected_sheet)
                        st.session_state.current_mode = "📊 数据分析"
                        st.success(f"Excel文件的工作表 '{selected_sheet}' 已成功加载！")
                    else:
                        st.error("Excel文件中没有找到任何工作表！")

        elif file_ext == "txt":
            content = uploaded_file.read().decode("utf-8")
            file_path = f"{session_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            st.session_state.txt_content = content
            st.session_state.session_id = session_id
            st.session_state.is_new_file = True
            st.session_state.current_mode = "📚 文档问答"
            st.success("文本文件已成功上传！")

        else:
            st.error("不支持的文件类型！")

    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")

# 初始化会话状态 (添加current_mode和selected_sheet)
def init_session_state():
    if 'current_session_messages' not in st.session_state:
        st.session_state.current_session_messages = [
            {'role': 'ai', 'content': '你好，我是你的AI助手，请问有什么能帮助你吗？'}]
    if 'history_sessions' not in st.session_state:
        st.session_state.history_sessions = []  # 存储所有历史会话
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'txt_content' not in st.session_state:
        st.session_state.txt_content = None
    if 'viewing_history' not in st.session_state:
        st.session_state.viewing_history = False
    if 'current_session_index' not in st.session_state:
        st.session_state.current_session_index = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if 'is_new_file' not in st.session_state:
        st.session_state.is_new_file = True
    if 'API_KEY' not in st.session_state:
        st.session_state.API_KEY = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"
    if 'model_temperature' not in st.session_state:
        st.session_state.model_temperature = 0.7
    if 'model_max_length' not in st.session_state:
        st.session_state.model_max_length = 1000
    if 'system_prompt' not in st.session_state:
        # 使用提示词
        st.session_state.system_prompt = TEXT_AGENT_PROMPT_TEMPLATE
    # 添加current_mode和selected_sheet状态
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "💬 聊天对话"
    if 'selected_sheet' not in st.session_state:
        st.session_state.selected_sheet = None

# 生成统计图表 (使用Plotly)
def create_chart(input_data, chart_type):
    """生成统计图表"""
    try:
        # 确保标题使用UTF-8编码
        title = input_data.get("title", "默认图表")
        if isinstance(title, str):
            title = title.encode('utf-8', 'ignore').decode('utf-8')

        # 检查数据格式 - 处理两种格式
        if isinstance(input_data["data"][0], list):
            # 处理二维数组格式 [["月份", 销售额], ...]
            df_data = pd.DataFrame(
                input_data["data"],
                columns=input_data["columns"]
            )
        else:
            # 处理一维数组格式 ["月份1", "月份2", ...] 和 [销售额1, 销售额2, ...]
            df_data = pd.DataFrame({
                input_data["columns"][0]: input_data["columns"],
                input_data["columns"][1]: input_data["data"]
            })

        # 确保所有列名都是字符串类型
        df_data.columns = df_data.columns.astype(str)

        # === 新增：输出图表数据到界面 ===
        st.subheader("📊 图表原始数据")
        st.dataframe(df_data, use_container_width=True)
        st.caption(f"数据维度: {df_data.shape[0]} 行 × {df_data.shape[1]} 列")
        # =============================

        # 根据图表类型生成不同的可视化
        if chart_type == "bar":
            fig = px.bar(
                df_data,
                x=df_data.columns[0],
                y=df_data.columns[1],
                title=input_data.get("title", "柱状图")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "line":
            fig = px.line(
                df_data,
                x=df_data.columns[0],
                y=df_data.columns[1],
                title=input_data.get("title", "折线图"),
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "pie":
            fig = px.pie(
                df_data,
                names=df_data.columns[0],
                values=df_data.columns[1],
                title=input_data.get("title", "饼图")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "scatter":
            fig = px.scatter(
                df_data,
                x=df_data.columns[0],
                y=df_data.columns[1],
                title=input_data.get("title", "散点图")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "box":
            fig = px.box(
                df_data,
                y=df_data.columns[1],
                title=input_data.get("title", "箱线图")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "hist":
            fig = px.histogram(
                df_data,
                x=df_data.columns[1],
                title=input_data.get("title", "直方图")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "area":
            fig = px.area(
                df_data,
                x=df_data.columns[0],
                y=df_data.columns[1],
                title=input_data.get("title", "面积图")
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"不支持的图表类型: {chart_type}")

    except Exception as e:
        st.error(f"图表生成出错：{e}")
        st.error(f"数据格式: {input_data}")

# 文本代理 - 处理无文件情况
def text_agent(query):
    try:
        model = ChatOpenAI(
            api_key=st.session_state.API_KEY,
            base_url='https://twapi.openai-hk.com/v1',
            model=st.session_state.selected_model,
            temperature=st.session_state.model_temperature,
            max_tokens=st.session_state.model_max_length
        )
        chain = ConversationChain(llm=model, memory=st.session_state.memory)
        return chain.invoke({'input': query})['response']
    except Exception as e:
        st.error(f"文本处理出错：{e}")
        return "无法处理您的请求，请检查配置或重试。"

# 提取图表数据
def extract_chart_data(response_text):
    """尝试从代理响应中提取图表数据"""
    try:
        # 查找JSON格式的图表数据
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        # 查找纯JSON格式
        elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
            return json.loads(response_text)
    except Exception as e:
        st.warning(f"图表数据解析失败: {str(e)}")
    return None

# 增强JSON解析能力
def safe_json_parse(response_text):
    """安全地解析可能包含额外内容的JSON字符串"""
    try:
        # 尝试直接解析整个响应
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # 尝试提取JSON部分
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass

    # 尝试提取JSON代码块
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
    except:
        pass

    # 尝试使用正则表达式提取JSON
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass

    return None


# 数据框代理 - 处理CSV/Excel文件
def dataframe_agent(df, query):
    try:
        # 使用提示词模板
        structured_prompt = DF_AGENT_PROMPT_TEMPLATE.format(
            df_head=df.head(3).to_string(),
            query=query
        )

        # 创建代理 - 显式设置响应编码
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                api_key=st.session_state.API_KEY,
                base_url='https://twapi.openai-hk.com/v1',
                model=st.session_state.selected_model,
                temperature=0.2,
                max_tokens=st.session_state.model_max_length,
                model_kwargs={'response_format': {'type': 'text'}}  # 确保响应是文本格式
            ),
            df,
            verbose=True,
            handle_parsing_errors=lambda _: "请按指定格式回复",
            max_iterations=3,
            allow_dangerous_code=True,
            include_df_in_prompt=True
        )

        # 获取代理响应并显式解码为UTF-8
        response = agent.invoke(structured_prompt)['output']

        # 显式编码为UTF-8
        if isinstance(response, str):
            response = response.encode('utf-8', 'ignore').decode('utf-8')

        # 调试信息
        st.toast(f"代理原始响应长度: {len(response)} 字符")

        # 尝试解析为结构化数据
        parsed_response = safe_json_parse(response)
        if parsed_response:
            return parsed_response

        # 修复：增强多行文本处理能力
        # 1. 检查是否包含多行数据（如每月销售额）
        if re.search(r'(月|月份|month).*[\d,]+', response, re.IGNORECASE):
            # 提取所有月份数据行
            lines = [line.strip() for line in response.split('\n') if re.search(r'[\d,]+', line)]

            # 格式化为更易读的形式
            formatted_response = "每月销售额统计：\n" + "\n".join(lines)
            return {"answer": formatted_response}

        # 2. 检查是否是单个数值答案
        if re.search(r'[\d,.]+', response):
            # 提取数值和单位
            match = re.search(r'([\d,.]+)(.*)', response)
            if match:
                value = match.group(1).replace(',', '')
                unit = match.group(2).strip()
                return {"answer": f"{value}{unit}"}

        # 3. 如果是普通文本，直接返回
        return {"answer": response}

    except Exception as e:
        st.error(f"数据分析出错：{e}")
        return {
            "error": str(e),
            "answer": "系统处理数据时出错"
        }

# RAG代理 - 处理TXT文件
def rag_agent(query):
    try:
        # 加载嵌入模型
        if 'em_model' not in st.session_state:
            try:
                # 使用正确的 BGE 中文模型
                model_name = "BAAI/bge-small-zh-v1.5"
                st.session_state.em_model = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    encode_kwargs={'normalize_embeddings': True}  # 重要参数
                )
            except Exception as e:
                st.error(f"加载嵌入模型失败：{e}")
                return {"answer": "无法加载文本处理模型，请检查网络连接"}

        # 如果是新文件，处理文本
        if st.session_state.is_new_file:
            with open(f'{st.session_state.session_id}.txt', 'w', encoding='utf-8') as f:
                f.write(st.session_state.txt_content)

            loader = TextLoader(f'{st.session_state.session_id}.txt', encoding='utf-8')
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                separators=["\n", "。", "！", "？", "，", "、", ""]
            )
            texts = text_splitter.split_documents(docs)

            db = FAISS.from_documents(texts, st.session_state.em_model)
            st.session_state.db = db
            st.session_state.is_new_file = False

        # 创建检索链
        model = ChatOpenAI(
            api_key=st.session_state.API_KEY,
            base_url='https://twapi.openai-hk.com/v1',
            model=st.session_state.selected_model,
            temperature=st.session_state.model_temperature,
            max_tokens=st.session_state.model_max_length
        )

        retriever = st.session_state.db.as_retriever()

        # 显式加载聊天历史
        chat_history = st.session_state.memory.load_memory_variables({})["history"]

        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            return_source_documents=True
        )

        # 传递 chat_history 参数
        result = chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        return {"answer": result['answer']}
    except Exception as e:
        st.error(f"文本处理出错：{e}")
        return {"answer": "无法处理文本内容，请重试或上传其他文件。"}

# 主应用
def main():
    # 设置页面配置（确保中文字符支持）
    st.set_page_config(
        page_title="SuperAI 智能分析助手",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # 页面标题
    header_container = st.container()
    with header_container:
        cols = st.columns([1, 8, 1])
        with cols[1]:
            st.markdown("""
                <div style="text-align:center; margin-bottom:40px">
                    <h1 style="margin-bottom:0">SuperAI 智能分析助手🚀</h1>
                    <p style="color:#6C63FF; font-size:1.2rem">数据洞察从未如此简单</p>
                </div>
            """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.title("超级智能分析助手")
        api_key = st.text_input('请输入OpenAI API Key', type='password', value=st.session_state.API_KEY)
        if api_key:
            st.session_state.API_KEY = api_key

        if st.button("🔄 新建会话", use_container_width=True):
            # 保存当前会话到历史会话
            if len(st.session_state.current_session_messages) > 1:  # 避免保存只有欢迎消息的会话
                new_session = {
                    'id': uuid.uuid4().hex,
                    'messages': st.session_state.current_session_messages.copy(),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M", time.localtime())
                }
                st.session_state.history_sessions.append(new_session)

            # 重置当前会话
            st.session_state.current_session_messages = [
                {'role': 'ai', 'content': '你好，我是你的AI助手，请问有什么能帮助你吗？'}]
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.session_state.viewing_history = False
            st.session_state.df = None
            st.session_state.txt_content = None
            st.session_state.is_new_file = True
            st.session_state.session_id = uuid.uuid4().hex
            # 新增：清除文件上传器状态
            st.session_state.file_uploader_key = str(uuid.uuid4())  # 生成新的随机键
            # 重置模式和工作表选择
            st.session_state.current_mode = "💬 聊天对话"
            st.session_state.selected_sheet = None
            st.rerun()

        st.divider()

        # 历史会话
        st.subheader("📜 历史会话")
        if st.session_state.history_sessions:
            for i, session in enumerate(st.session_state.history_sessions):
                # 查找第一条用户消息作为预览
                user_preview = ""
                for msg in session['messages']:
                    if msg['role'] == 'human':
                        user_preview = msg['content'][:30] + ('...' if len(msg['content']) > 30 else '')
                        break

                st.caption(f"📅 {session['timestamp']}")
                st.caption(f"🗣️ 用户: {user_preview}")

                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"查看会话 {i + 1}", key=f"view_{i}", use_container_width=True):
                        st.session_state.viewing_history = True
                        st.session_state.current_session_index = i
                with col2:
                    if st.button("❌", key=f"delete_{i}", use_container_width=True):
                        del st.session_state.history_sessions[i]
                        st.rerun()
                st.divider()
        else:
            st.caption("暂无历史会话")
        st.divider()

        # 模型配置
        st.subheader("⚙️ 模型配置")
        st.session_state.selected_model = st.selectbox(
            "选择AI模型",
            ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=1,
            help="选择要使用的AI模型"
        )

        st.session_state.model_temperature = st.slider(
            "温度 (Temperature)",
            0.0, 1.0, 0.7, 0.1,
            help="控制生成文本的随机性，值越高越有创意，值越低越稳定"
        )

        st.session_state.model_max_length = st.slider(
            "最大生成长度",
            100, 4000, 1000, 100,
            help="限制AI生成的最大token数量"
        )

        st.session_state.system_prompt = st.text_area(
            "系统提示词",
            # 使用提示词
            RAG_AGENT_PROMPT_TEMPLATE,
            help="指导AI如何回答问题的系统级提示"
        )

    # 查看历史会话
    if st.session_state.viewing_history and st.session_state.current_session_index is not None:
        st.subheader("📜 历史消息")
        session = st.session_state.history_sessions[st.session_state.current_session_index]

        for message in session['messages']:
            with st.chat_message("user" if message["role"] == "human" else "assistant"):
                st.write(message["content"])

        if st.button("↩️ 返回当前对话", use_container_width=True):
            st.session_state.viewing_history = False
            st.rerun()

    # 主界面 - 文件上传和聊天
    else:
        # 文件上传区域
        st.subheader("📤 上传数据文件")
        file = st.file_uploader(
            "上传CSV、Excel或TXT文件",
            type=["csv", "xlsx", "txt"],
            label_visibility="collapsed",
            key=st.session_state.get('file_uploader_key', 'default_file_uploader')
        )

        # 处理文件上传 - 使用从app.py添加的process_uploaded_file函数
        if file:
            process_uploaded_file(file)

        # 显示当前模式 (从app.py添加)
        if 'current_mode' in st.session_state:
            st.markdown(f"**当前模式**: {st.session_state.current_mode}")

        # 重置文件状态逻辑
        if file is None and (st.session_state.df is not None or st.session_state.txt_content is not None):
            # 用户已删除文件，重置相关状态
            st.session_state.df = None
            st.session_state.txt_content = None
            st.session_state.is_new_file = True
            st.toast("文件已移除，现在可进行文本问答")
            # 清除预览区域
            st.rerun()
        elif file:
            # 处理文件上传后显示预览
            try:
                file_type = file.name.split('.')[-1].lower()
                if file_type in ['csv', 'xlsx'] and st.session_state.df is not None:
                    with st.expander("👀 数据预览", expanded=True):
                        st.dataframe(st.session_state.df.head(10), use_container_width=True)
                        st.caption(f"数据维度: {st.session_state.df.shape[0]} 行 × {st.session_state.df.shape[1]} 列")
                elif file_type == 'txt' and st.session_state.txt_content is not None:
                    with st.expander("📝 文本内容预览", expanded=True):
                        st.text_area("", st.session_state.txt_content, height=300, label_visibility="collapsed")
            except Exception as e:
                st.error(f"文件预览错误: {str(e)}")

        # 显示当前会话聊天历史
        for message in st.session_state.current_session_messages:
            with st.chat_message("user" if message["role"] == "human" else "assistant"):
                st.write(message["content"])

        # 用户输入
        if prompt := st.chat_input("请输入您的问题...", key="user_input"):
            if not st.session_state.API_KEY:
                st.error('🔑 请输入OpenAI API Key')
                st.stop()

            # 添加用户消息到当前会话
            st.session_state.current_session_messages.append({'role': 'human', 'content': prompt})

            with st.chat_message("user"):
                st.write(prompt)

            # AI处理区域
            with st.spinner('🤖 AI正在思考，请稍等...'):
                try:
                    # 根据文件类型选择处理方式
                    if st.session_state.df is not None:
                        response = dataframe_agent(st.session_state.df, prompt)
                    elif st.session_state.txt_content is not None:
                        response = rag_agent(prompt)
                    else:
                        # 没有文件时使用文本代理
                        response = {"answer": text_agent(prompt)}

                    # 确保response是字典类型
                    if not isinstance(response, dict):
                        response = {"answer": str(response)}

                    # 处理错误响应
                    if "error" in response:
                        st.error(f"错误: {response['error']}")
                        ai_response = response.get("answer", "数据分析失败")
                    else:
                        # 提取文本回答
                        ai_response = response.get("answer", "没有获取到回答内容")

                        # 移除不需要的文本
                        unwanted_phrases = [
                            "详细的文本解释和统计结果:",
                            "详细的文本解释和统计结果：",
                            "详细的文本解释和统计结果:",
                            "详细解释和统计结果:",
                            "🤖"
                        ]

                        for phrase in unwanted_phrases:
                            ai_response = ai_response.replace(phrase, "").strip()

                        # 图表关键词列表
                        chart_keywords = ["图表", "柱状图", "折线图", "饼图", "可视化", "展示图", "散点图", "箱线图",
                                          "直方图", "面积图"]

                        # 只在用户明确要求图表且response是字典时才检查
                        if isinstance(response, dict) and "charts" in response:
                            if any(kw in prompt.lower() for kw in chart_keywords):
                                # === 新增：输出图表标题和类型 ===
                                st.subheader("📈 图表信息")
                                st.markdown(f"**图表类型**: {response['charts'][0]['type']}")
                                st.markdown(f"**图表标题**: {response['charts'][0].get('title', '无标题')}")

                                for chart in response["charts"]:
                                    if "type" in chart and "data" in chart:
                                        create_chart(chart["data"], chart["type"])
                        else:
                            # 移除可能的图表生成消息
                            ai_response = ai_response.split("\n\n已生成")[0]

                    # 添加AI响应到当前会话
                    st.session_state.current_session_messages.append({'role': 'ai', 'content': ai_response})

                    # 显示AI响应
                    with st.chat_message("assistant"):
                        if "error" in response:
                            st.error(ai_response)
                        else:
                            st.write(ai_response)
                except Exception as e:
                    error_msg = f"处理请求时出错: {str(e)}"
                    st.session_state.current_session_messages.append({'role': 'ai', 'content': error_msg})
                    with st.chat_message("assistant"):
                        st.error(error_msg)


if __name__ == "__main__":
    main()