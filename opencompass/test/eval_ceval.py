from mmengine.config import read_base
from opencompass.models import OpenAI
with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

# Only test ceval-computer_network dataset in this demo
datasets = ceval_datasets[:1]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

VLLM_API_BASE = 'http://192.168.1.212:9401/v1/chat/completions'  # 完整路径（推荐）
# VLLM_API_BASE = 'http://192.168.1.212:9401/v1'  # 基础路径（如果上面不工作，尝试这个）
MODEL_NAME = ''  # 修改为你的模型名称（仅用于标识，不能为空）
# meta_template = dict(
#     round=[
#         dict(role='HUMAN', api_role='user', begin='[[HUMAN]]\n', end='\n'),
#         dict(role='BOT', api_role='assistant', begin='[[BOT]]\n', end='\n', generate=True)
#     ],
# )
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='system')],
)

# 修正后的 API 地址和模型标识
VLLM_API_BASE = 'http://192.168.1.212:9401/v1'
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