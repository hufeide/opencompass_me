"""
OpenCompass vLLM API 评估配置
基于 /data/hxf/bigmodel/opencompass/opencompass/configs/eval_chinese_vllm.py 的配置方式

使用方法:
    python run.py eval_vllm_api.py --debug
"""

from mmengine.config import read_base
from opencompass.models import OpenAI

# ============= 配置区域：请根据实际情况修改 =============

# vLLM API 服务地址
VLLM_API_BASE = 'http://192.168.1.212:8056/v1'  # 请修改为你的 vLLM API 地址

# 模型标识名称（仅用于结果标识）
MODEL_ABBR = 'your-vllm-model'

# 数据集配置
# 可以导入多个数据集进行评估
with read_base():
    # 示例：导入中文评估数据集
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # 示例：导入英文数学推理数据集
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets

# 选择要评估的数据集
datasets = [*ceval_datasets]  # 使用中文评测数据集
# datasets = gsm8k_datasets + math_datasets  # 或使用英文数学推理数据集
# datasets = [*ceval_datasets] + gsm8k_datasets + math_datasets  # 或组合多个数据集

# 可选：统一修改数据集标识（例如添加前缀）
# for dataset in datasets:
#     dataset['abbr'] = 'MS_' + dataset['abbr']

# ============= 模型配置 =============
# meta_template 需要根据你的 vLLM 部署的模型对话格式进行配置

# 配置1: 标准 OpenAI ChatML 格式（大多数情况下使用这个）
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
)

# 配置2: 自定义格式（如参考文件中的格式）
# meta_template = dict(
#     round=[
#         dict(role='HUMAN', begin='[[HUMAN]]\n', end='\n'),
#         dict(role='BOT', begin='[[BOT]]\n', end='\n', generate=True)
#     ],
# )

models = [
    dict(
        type=OpenAI,
        abbr=MODEL_ABBR,
        path='your-model-name',  # 模型名称或路径（用于标识，实际调用使用 API）
        key='EMPTY',  # vLLM API 通常不需要 key
        openai_api_base=VLLM_API_BASE,
        meta_template=api_meta_template,  # 使用标准格式，或替换为上面的自定义格式
        query_per_second=10,  # 每秒请求数限制（根据服务器性能调整，避免过载）
        max_out_len=2048,  # 最大输出 token 数
        max_seq_len=4096,  # 最大序列长度（prompt + output 总长度）
        batch_size=8,  # 批处理大小
        mode='chat',  # 模式：'chat' 用于对话模型，'none' 用于基础模型
    )
]

# ============= 如何确定 meta_template =============
# 
# meta_template 的格式取决于你的 vLLM 部署时使用的模板格式。
# 
# 常见格式：
# 1. OpenAI ChatML 格式（最常用）：
#    - api_role='user' 和 api_role='assistant'
# 
# 2. 自定义格式：
#    - 使用 begin 和 end 定义每个角色的开始和结束标记
#    - 例如：begin='[[HUMAN]]\n', end='\n'
# 
# 如果不确定，可以：
# 1. 查看 vLLM 部署的配置文件
# 2. 测试不同的格式，看哪种能正常工作
# 3. 参考 /data/hxf/bigmodel/opencompass/opencompass/configs/eval_chinese_vllm.py

