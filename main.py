"""读取已有 JSON 数据并通过百炼生成总结。"""
import argparse
import json
import logging
from pathlib import Path

from config import Settings
from llm.qwen_client import QwenSummarizer


def main():
    """从命令行读取 JSON 文件，调用模型并将总结打印到终端。"""
    parser = argparse.ArgumentParser(description='使用百炼 Qwen 总结 JSON 数据')
    parser.add_argument('data_file', type=Path, help='待分析的 UTF-8 JSON 文件路径')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # utf-8-sig 也能读取带 BOM 的 UTF-8 文件。
    with args.data_file.open(encoding='utf-8-sig') as file:
        data = json.load(file)
    # 先加载配置，再通过百炼接口生成总结。
    summarizer = QwenSummarizer.from_settings(Settings.load())
    print(summarizer.summarize(data))


if __name__ == '__main__':
    main()
