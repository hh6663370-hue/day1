"""用 LangChain 和本地 Transformers 模型总结已准备好的数据。"""
import logging
import time
from pathlib import Path

from llm.prompt import build_user_prompt, load_system_prompt

logger = logging.getLogger(__name__)


def build_chain(model_path: Path, max_new_tokens: int = 2048):
    """加载本地模型，并组装 Jinja 提示词和 Transformers 推理组件。"""
    if not model_path.is_dir():
        raise FileNotFoundError(f'模型目录不存在: {model_path}')
    from langchain_core.runnables import RunnableLambda
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # 只从本地目录读取权重，并自动分配设备和模型精度。
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, device_map='auto', torch_dtype='auto'
    )
    generator = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False,
    )
    prompt = build_user_prompt()
    system_prompt = load_system_prompt()

    def format_chat(value):
        # 使用 Qwen 自带的模板构造带角色的对话。
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': value.to_string()}],
            tokenize=False, add_generation_prompt=True,
        )

    return prompt | RunnableLambda(format_chat) | HuggingFacePipeline(pipeline=generator)


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
        """根据配置创建模型链；模型加载失败会直接向上抛出。"""
        # 模型只加载一次；调用失败时只重试生成，不重复加载模型。
        chain = build_chain(settings.model_path, settings.max_new_tokens)
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
