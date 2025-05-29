import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量（只需一次）
load_dotenv()

# 数据框代理提示词模板 - 优化版
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
    "answer": "详细文本解释(解释用户问题)和统计结果:",
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


def dataframe_agent(query, df_head):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        base_url='https://twapi.openai-hk.com/v1',  # 统一API端点
        temperature=0
    )
    # 使用提示词模板
    prompt = DF_AGENT_PROMPT_TEMPLATE.format(df_head=df_head, query=query)

    try:
        response = model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response

        # 尝试解析为JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)

            # 作为最后手段返回纯文本
            return {"answer": content}

    except Exception as e:
        return {"error": str(e), "answer": "请求处理失败"}


def text_agent(query):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        base_url='https://twapi.openai-hk.com/v1',  # 统一API端点
        temperature=0.7,
        max_tokens=1000
    )
    prompt = f"你是一个乐于助人的AI助手，用中文回答问题。问题：{query}"

    try:
        response = model.invoke(prompt)
        return response.content if hasattr(response, 'content') else response
    except Exception as e:
        return f"请求处理失败: {str(e)}"