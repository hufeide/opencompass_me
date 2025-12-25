from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # 1. 将相对导入改为绝对导入（去掉前面的点）
    from opencompass.configs.summarizers.medium import summarizer
    
    # 2. 确保 ceval 导入也是完整路径
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

datasets = [
*ceval_datasets,
]

api_meta_template = dict(
round=[
dict(role='HUMAN', api_role='HUMAN'),
dict(role='BOT', api_role='BOT', generate=True),
],
)

models = [
dict(
abbr='',
type=OpenAI,
path='qwen3_next_80B',
key='qwen3_next_80B', # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
openai_api_base='http://192.168.1.212:9401/v1/chat/completions',
max_seq_len=8000,
meta_template=api_meta_template,
query_per_second=1,
max_out_len=8000,
batch_size=6,

    #run_cfg=dict(num_gpus=0),
    ),
]

infer = dict(
partitioner=dict(type=NaivePartitioner),
runner=dict(
type=LocalRunner,
max_num_workers=4,
task=dict(type=OpenICLInferTask)),
)