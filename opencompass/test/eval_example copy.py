"""
OpenCompass 评估示例脚本 - vLLM API 版本

使用方法:
1. 如果使用 ceval 数据集，建议使用 ModelScope 模式（自动下载数据）:
   export DATASET_SOURCE=ModelScope
   python run.py eval_example.py --debug

2. 或者先使用 demo 数据集测试 API 连接（不需要下载数据）

3. 修改下方配置:
   - VLLM_API_BASE: vLLM API 服务地址
   - MODEL_NAME: 模型名称（用于标识）
   - meta_template: 根据你的模型对话格式进行调整

4. 查看所有可用数据集:
   python tools/list_configs.py | grep dataset
"""

from mmengine.config import read_base
from opencompass.models import OpenAI

# 1. 导入数据集配置
# 注意：请务必确认下方文件名与您磁盘上的实际文件名一致（带不带哈希后缀）
with read_base():
    # 选项1: 使用 ceval 数据集（需要设置 DATASET_SOURCE=ModelScope 或手动下载数据）
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    
    # 选项2: 使用 demo 数据集（推荐用于首次测试，不需要下载数据）
    # from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    # from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets

# 2. 定义数据集
# 选项1: 使用 ceval 数据集
datasets = [*ceval_datasets]

# 选项2: 使用 demo 数据集（推荐用于首次测试）
# datasets = gsm8k_datasets + math_datasets

# 如果需要组合多个数据集，可以这样：
# datasets = [*ceval_datasets] + gsm8k_datasets + math_datasets

# 可选：统一修改数据集标识（例如添加前缀，参考 eval_chinese_vllm.py）
# for dataset in datasets:
#     dataset['abbr'] = 'MS_' + dataset['abbr']

# 3. 配置 vLLM API 模型
# ============= 请根据你的实际情况修改以下配置 =============
# vLLM API 地址配置
# 注意：通常需要包含完整的端点路径
# 如果使用完整路径不工作，可以尝试只使用基础路径（如参考文件中的格式）
VLLM_API_BASE = 'http://192.168.1.212:9401/v1/chat/completions'  # 完整路径（推荐）
# VLLM_API_BASE = 'http://192.168.1.212:9401/v1'  # 基础路径（如果上面不工作，尝试这个）
MODEL_NAME = ''  # 修改为你的模型名称（仅用于标识，不能为空）

# meta_template 配置需要根据你的模型对话格式进行调整
# 以下是几种常见的格式示例：

# 示例1: 标准对话格式（类似 OpenAI ChatML 格式）
# meta_template = dict(
#     round=[
#         dict(role='HUMAN', api_role='user'),  # 使用 api_role 映射到 API 的 role
#         dict(role='BOT', api_role='assistant', generate=True),
#     ],
# )

# 示例2: 自定义格式（如参考文件中的格式）
# 注意：如果使用 begin/end，对于 API 模型通常需要同时提供 api_role
# 对于标准 OpenAI API 格式，推荐使用示例1的 api_role 方式
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user', begin='[[HUMAN]]\n', end='\n'),
        dict(role='BOT', api_role='assistant', begin='[[BOT]]\n', end='\n', generate=True)
    ],
)

# 如果上面的格式（begin/end + api_role）不工作，可以尝试纯 api_role 格式（推荐用于标准 OpenAI API）：
# api_meta_template = dict(
#     round=[
#         dict(role='HUMAN', api_role='user'),
#         dict(role='BOT', api_role='assistant', generate=True),
#     ],
# )
# meta_template = api_meta_template

# ============================================================

models = [
    dict(
        type=OpenAI,
        abbr=MODEL_NAME,  # 模型标识名称
        path='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8',  # 对于 vLLM API，这个值通常可以设置为模型名称或路径
        key='EMPTY',  # vLLM API 通常不需要 key，设置为 'EMPTY'
        openai_api_base=VLLM_API_BASE,  # vLLM API 服务地址
        meta_template=meta_template,  # 对话模板配置
        query_per_second=10,  # 每秒请求数限制（根据服务器性能调整）
        max_out_len=2048,  # 最大输出长度
        max_seq_len=4096,  # 最大序列长度（prompt + output）
        batch_size=8,  # 批处理大小
        mode='none',  # 模式：'none'（不截断）、'front'（保留后面）、'mid'（保留中间）、'rear'（保留前面）
    )
]

