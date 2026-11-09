"""从项目 .env 及环境变量加载本地模型配置。"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """保存模型路径、最大调用次数、重试间隔和生成长度。"""
    model_path: Path
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_new_tokens: int = 2048

    @classmethod
    def load(cls):
        """读取项目配置；系统环境变量优先于 .env 中的同名配置。"""
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
        model_path = os.getenv('QWEN_MODEL_PATH', '').strip()
        if not model_path:
            raise ValueError('请在 .env 中配置 QWEN_MODEL_PATH')
        # 最大调用次数包含首次调用；可通过 .env 自行设置。
        attempts = int(os.getenv('QWEN_MAX_ATTEMPTS', '3'))
        delay = float(os.getenv('QWEN_RETRY_DELAY_SECONDS', '2'))
        tokens = int(os.getenv('QWEN_MAX_NEW_TOKENS', '2048'))
        # 在加载大型模型之前检查参数，避免无效配置导致资源浪费。
        if attempts < 1 or delay < 0 or tokens < 1:
            raise ValueError('调用次数和输出 token 数须大于 0，重试等待不能为负')
        return cls(Path(model_path), attempts, delay, tokens)
