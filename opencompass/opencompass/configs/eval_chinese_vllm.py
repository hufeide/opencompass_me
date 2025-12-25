from mmengine.config import read_base
from opencompass.models import OpenAI

# 1. 基础配置导入
# 注意：请务必确认下方文件名与您磁盘上的实际文件名一致（带不带哈希后缀）
with read_base():
    # 如果您的目录下是 ceval_gen_5fa3db.py，请保留后缀
    # 如果您的目录下是 ceval_gen.py，请去掉后缀
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets

# 2. 定义数据集
datasets = [*ceval_datasets]
# /home/aixz/data/hxf/bigmodel/opencompass/opencompass/datasets/ceval/ceval_gen_5f30c7.py
# /home/aixz/data/hxf/bigmodel/opencompass/opencompass/configs/datasets/ceval/ceval_gen_5f30c7.py

# 统一修改 abbr (ModelScope 标识)
for dataset in datasets:
    dataset['abbr'] = 'MS_' + dataset['abbr']

# 3. 定义模型 (vLLM)
models = [
    dict(
        type=OpenAI,
        abbr='NVIDIA-Nemotron-Nano-vLLM',
        path='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8', 
        key='EMPTY', 
        openai_api_base='http://192.168.1.212:8056/v1', 
        meta_template=dict(
            round=[
                dict(role='HUMAN', begin='[[HUMAN]]\n', end='\n'),
                dict(role='BOT', begin='[[BOT]]\n', end='\n', generate=True)
            ],
        ),
        query_per_second=10, 
        max_out_len=1024,
        batch_size=10,
        mode='chat'
    )
]


