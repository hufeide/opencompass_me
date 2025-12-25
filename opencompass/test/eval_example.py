from mmengine.config import read_base
from opencompass.datasets.circular import CircularCEvalDataset, CircularEvaluator
from opencompass.summarizers import CircularSummarizer
from mmengine.config import read_base
from opencompass.models import OpenAI

with read_base():
    # 只保留 ceval 数据集配置
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # 只保留 ceval 的汇总组配置
    from opencompass.configs.summarizers.groups.ceval import ceval_summary_groups

# 1. 配置 Circular Eval 逻辑
for d in ceval_datasets:
    d['type'] = CircularCEvalDataset
    d['abbr'] = d['abbr'] + '-circular-4'
    d['eval_cfg']['evaluator'] = {
        'type': CircularEvaluator,
        'circular_pattern': 'circular'
    }
    d['circular_patterns'] = 'circular'

datasets = ceval_datasets

# 2. 配置 Summarizer
# 生成带有 circular 后缀的新汇总组
new_summary_groups = []
for item in ceval_summary_groups:
    new_summary_groups.append({
        'name': item['name'] + '-circular-4',
        'subsets': [i + '-circular-4' for i in item['subsets']],
    })

summarizer = dict(
    type=CircularSummarizer,
    metric_types=['acc_origin', 'perf_circular'],
    dataset_abbrs=[
        'ceval-circular-4',
        'ceval-humanities-circular-4',
        'ceval-stem-circular-4',
        'ceval-social-science-circular-4',
        'ceval-other-circular-4',
    ],
    summary_groups=new_summary_groups,
)

VLLM_API_BASE = 'http://192.168.1.212:9401/v1/chat/completions'  # 完整路径（推荐）
# VLLM_API_BASE = 'http://192.168.1.212:9401/v1'  # 基础路径（如果上面不工作，尝试这个）
MODEL_NAME = ''  # 修改为你的模型名称（仅用于标识，不能为空）

# meta_template 配置需要根据你的模型对话格式进行调整
# 以下是几种常见的格式示例：


# 示例2: 自定义格式（如参考文件中的格式）
# 注意：如果使用 begin/end，对于 API 模型通常需要同时提供 api_role
# 对于标准 OpenAI API 格式，推荐使用示例1的 api_role 方式
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user', begin='[[HUMAN]]\n', end='\n'),
        dict(role='BOT', api_role='assistant', begin='[[BOT]]\n', end='\n', generate=True)
    ],
)


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

