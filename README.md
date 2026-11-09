# 本地换机数据分析

程序通过 LangChain、Jinja 模板和 Transformers，将已有的换机统计数据交给本地 Qwen2.5-7B-Instruct 模型生成报告。统计指标及异常由上游提供，程序不计算这些值。

安装依赖：`uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt`。

编辑 `.env` 中的 `QWEN_MODEL_PATH`、`QWEN_MAX_ATTEMPTS` 和 `QWEN_MAX_NEW_TOKENS`。`QWEN_MODEL_PATH` 指向本地模型目录。最大调用次数包含首次调用；生成失败才重试，模型加载失败直接报错。本地加载不使用 API 地址或 API 超时设置。

输入文件必须是 JSON 对象，字段格式见 `tests/sample_records.json`：
- `start`、`end`、`days`、`total`、`tickets`、`people`、`avg_per_day`、`peak_date`、`peak_value`、`repeat_threshold`、`repeat_count` 为已计算好的指标。
- `daily` 为按日统计对象列表，每项包含 `date`、`count`、`tickets`、`people`。
- `people_top5` 为人员列表，每项包含 `name`、`count`、`percentage`；百分比由上游提供，例如 `"50%"`。
- `devices_top5` 为设备列表，每项包含 `sn`、`count`、`tickets`。
- `anomalies` 为上游已判定的异常描述字符串列表。

未提供的指标或空列表会显示“数据不足”。程序不检查指标之间是否一致，也不自动截断长输入。提示词要求报告为 600–900 字和六个标题，但模型输出不保证完全遵循。

运行：` .\.venv\Scripts\python.exe main.py tests\sample_records.json`。

测试：` .\.venv\Scripts\python.exe -m unittest discover -s tests -v`。

系统提示词和 Jinja 用户提示词分别保存在 `llm/prompt/system_prompt.txt` 和 `llm/prompt/user_prompt.txt`。修改后需重新创建模型调用链或重新运行程序。

## NVIDIA 显卡运行

本机提供 `.venv-cuda`，使用 CUDA 12.8 版 PyTorch。运行示例：

```powershell
.\.venv-cuda\Scripts\python.exe main.py tests\sample_records.json
```

如需重新创建 GPU 环境：

```powershell
uv venv .venv-cuda --python 3.14
uv pip install --python .\.venv-cuda\Scripts\python.exe -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

本机显存不足以容纳完整模型，`device_map=auto` 会将部分模型放在显卡、其余放在系统内存中，运行速度受 CPU 和内存带宽影响。`.venv` 与 `.venv-cuda` 是独立环境。

