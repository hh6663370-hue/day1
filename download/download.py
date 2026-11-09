from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=r"D:\Tools\models\Qwen2.5-7B-Instruct"
)
print("模型下载完成，保存路径：", model_dir)
