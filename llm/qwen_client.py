"""使用 LangChain 提示词和百炼 OpenAI 兼容接口总结数据。"""
import logging
import time

from llm.prompt import build_user_prompt, load_system_prompt

logger = logging.getLogger(__name__)


class CloudChain:
    """通过流式接口收集最终回答，不输出思考过程。"""
    def __init__(self, client, model, system_prompt, prompt, max_tokens=2048):
        self.client, self.model = client, model
        self.system_prompt, self.prompt = system_prompt, prompt
        self.max_tokens = max_tokens

    def invoke(self, values):
        user_prompt = self.prompt.format(**values)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_prompt},
                      {'role': 'user', 'content': user_prompt}],
            extra_body={'enable_thinking': False}, stream=True,
            max_tokens=self.max_tokens,
        )
        pieces = []
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, 'finish_reason', None) == 'length':
                raise ValueError('模型输出被长度限制截断，请增大 QWEN_MAX_NEW_TOKENS')
            text = choice.delta.content
            if text:
                pieces.append(text)
        return ''.join(pieces)


def build_chain(settings):
    """建立百炼客户端并加载 LangChain Jinja 用户模板。"""
    from openai import OpenAI
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                    timeout=settings.timeout, max_retries=0)
    return CloudChain(client, settings.model, load_system_prompt(),
                      build_user_prompt(), settings.max_new_tokens)


class QwenSummarizer:
    """接收分析数据，并对模型生成失败执行有限次数重试。"""
    def __init__(self, chain, max_attempts=3, retry_delay=2.0):
        if max_attempts < 1 or retry_delay < 0:
            raise ValueError('调用次数须大于 0，重试等待不能为负')
        self.chain = chain
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    @classmethod
    def from_settings(cls, settings):
        """根据配置创建百炼调用链。"""
        chain = build_chain(settings)
        return cls(chain, settings.max_attempts, settings.retry_delay)

    def summarize(self, data):
        """根据已有统计指标生成中文总结，失败时按配置重试。"""
        if not data:
            raise ValueError('没有可分析的数据')
        if not isinstance(data, dict):
            raise ValueError('分析数据必须是包含统计字段的 JSON 对象')
        # 缺失指标明确标记为数据不足；缺失列表留空，由模板说明。
        scalar_fields = ('start', 'end', 'days', 'total', 'tickets', 'people',
                         'avg_per_day', 'peak_date', 'peak_value',
                         'repeat_threshold', 'repeat_count')
        list_fields = ('daily', 'people_top5', 'devices_top5', 'anomalies')
        payload = {key: data.get(key) if data.get(key) is not None else '数据不足'
                   for key in scalar_fields}
        for key in list_fields:
            value = data.get(key)
            if value is not None and not isinstance(value, list):
                raise ValueError(f'{key} 必须是列表')
            payload[key] = value or []
        # 最大调用次数包含第一次调用；空响应也作为失败处理。
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.chain.invoke(payload)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError('模型返回了空结果')
                return result.strip()
            except Exception as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(f'模型调用失败，已尝试 {attempt} 次') from exc
                # 保留原始异常作为最终错误的原因，便于排查。
                logger.warning('模型调用第 %s 次失败，准备重试：%s', attempt, exc)
                time.sleep(self.retry_delay)
