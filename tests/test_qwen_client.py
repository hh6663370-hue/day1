import unittest
import os
import tempfile
from unittest.mock import patch

from config import Settings
from llm.qwen_client import QwenSummarizer
from llm.prompt import load_system_prompt, load_user_prompt
from jinja2.sandbox import SandboxedEnvironment


class FakeChain:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def invoke(self, value):
        self.calls.append(value)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class SummarizerTests(unittest.TestCase):
    def test_serializes_chinese_data_and_returns_summary(self):
        chain = FakeChain(['总结结果'])
        client = QwenSummarizer(chain=chain, retry_delay=0)
        self.assertEqual(client.summarize({'total': 1, 'people_top5': [{'name': '张三', 'count': 1, 'percentage': '100%'}]}), '总结结果')
        self.assertEqual(chain.calls[0]['people_top5'][0]['name'], '张三')

    def test_retries_twice_then_succeeds(self):
        chain = FakeChain([RuntimeError('失败'), RuntimeError('失败'), '成功'])
        client = QwenSummarizer(chain=chain, retry_delay=0)
        self.assertEqual(client.summarize({'total': 1}), '成功')
        self.assertEqual(len(chain.calls), 3)

    def test_stops_after_three_failures(self):
        chain = FakeChain([RuntimeError('失败')] * 3)
        client = QwenSummarizer(chain=chain, retry_delay=0)
        with self.assertRaisesRegex(RuntimeError, '3'):
            client.summarize({'total': 1})
        self.assertEqual(len(chain.calls), 3)

    def test_environment_controls_maximum_attempts(self):
        with patch.dict('os.environ', {'QWEN_MAX_ATTEMPTS': '5', 'QWEN_RETRY_DELAY_SECONDS': '0'}):
            settings = Settings.load()
        chain = FakeChain([RuntimeError('失败')] * 4 + ['成功'])
        client = QwenSummarizer(chain, settings.max_attempts, settings.retry_delay)
        self.assertEqual(client.summarize({'total': 1}), '成功')
        self.assertEqual(len(chain.calls), 5)

    def test_model_chain_receives_original_model_directory(self):
        from pathlib import Path
        settings = Settings(Path('original-model'), max_new_tokens=512)
        with patch('llm.qwen_client.build_chain', return_value=FakeChain(['ok'])) as build:
            client = QwenSummarizer.from_settings(settings)
        build.assert_called_once_with(Path('original-model'), 512)
        self.assertEqual(client.summarize({'total': 1}), 'ok')

    def test_prompt_files_load_from_any_working_directory(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertIn('设备换机数据分析师', load_system_prompt())
                self.assertIn('{{ start }}', load_user_prompt())
            finally:
                os.chdir(previous)

    def test_jinja_renders_provided_statistics_and_missing_values(self):
        prompt = SandboxedEnvironment().from_string(load_user_prompt())
        text = prompt.render(start='2026-10-01', end='2026-10-02', days=2,
                             total=2, tickets=2, people=1, avg_per_day=1,
                             peak_date='2026-10-01', peak_value=2,
                             repeat_threshold=2, repeat_count=0,
                             daily=[{'date': '2026-10-01', 'count': 2, 'tickets': 2, 'people': 1}],
                             people_top5=[{'name': '张三', 'count': 2, 'percentage': '100%'}],
                             devices_top5=[], anomalies=[])
        self.assertIn('2026-10-01 至 2026-10-02，共 2 天', text)
        self.assertIn('张三', text)
        self.assertIn('数据不足', text)
        self.assertNotIn('{{', text)
        self.assertIn('## 六、改进建议', text)

    def test_missing_fields_are_passed_as_data_insufficient(self):
        chain = FakeChain(['总结'])
        QwenSummarizer(chain, retry_delay=0).summarize({'total': 10})
        self.assertEqual(chain.calls[0]['total'], 10)
        self.assertEqual(chain.calls[0]['peak_date'], '数据不足')
        self.assertEqual(chain.calls[0]['daily'], [])

    def test_nonpositive_attempts_are_rejected(self):
        with patch.dict('os.environ', {'QWEN_MAX_ATTEMPTS': '0'}):
            with self.assertRaises(ValueError):
                Settings.load()
        with self.assertRaises(ValueError):
            QwenSummarizer(FakeChain([]), max_attempts=0)

    def test_rejects_empty_data(self):
        client = QwenSummarizer(chain=FakeChain([]), retry_delay=0)
        with self.assertRaises(ValueError):
            client.summarize([])


if __name__ == '__main__':
    unittest.main()
