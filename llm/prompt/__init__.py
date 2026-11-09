"""从本目录加载系统提示词和 Jinja 用户模板。"""
from pathlib import Path


_PROMPT_DIR = Path(__file__).resolve().parent


def load_system_prompt():
    """读取 UTF-8 格式的系统提示词。"""
    return (_PROMPT_DIR / 'system_prompt.txt').read_text(encoding='utf-8-sig')


def load_user_prompt():
    """读取 UTF-8 格式的 Jinja 用户提示词。"""
    return (_PROMPT_DIR / 'user_prompt.txt').read_text(encoding='utf-8-sig')


def build_user_prompt():
    """从文本文件构建 LangChain 的 Jinja 提示词模板。"""
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate.from_template(load_user_prompt(), template_format='jinja2')
