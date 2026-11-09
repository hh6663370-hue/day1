"""从项目 .env 和环境变量加载百炼调用配置。"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_new_tokens: int = 2048
    timeout: float = 120.0

    @classmethod
    def load(cls):
        """系统环境变量优先于 .env；密钥不会出现在配置对象的 repr 中。"""
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
        api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
        if not api_key:
            raise ValueError('请在 .env 中配置 DASHSCOPE_API_KEY')
        url = os.getenv('DASHSCOPE_BASE_URL', '').strip()
        model = os.getenv('DASHSCOPE_MODEL', 'qwen3.8-27b').strip()
        attempts = int(os.getenv('QWEN_MAX_ATTEMPTS', '3'))
        delay = float(os.getenv('QWEN_RETRY_DELAY_SECONDS', '2'))
        tokens = int(os.getenv('QWEN_MAX_NEW_TOKENS', '2048'))
        timeout = float(os.getenv('DASHSCOPE_TIMEOUT_SECONDS', '120'))
        if not url or not model or attempts < 1 or delay < 0 or tokens < 1 or timeout <= 0:
            raise ValueError('请检查百炼地址、模型名、重试次数、间隔、输出长度和超时配置')
        return cls(api_key, url, model, attempts, delay, tokens, timeout)
