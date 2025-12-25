#!/bin/bash
# OpenCompass 评估运行脚本示例

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_google

# 切换到工作目录
cd /data/hxf/bigmodel/opencompass

# 设置默认参数
WORK_DIR="outputs/evaluation"
DATASETS="demo_gsm8k_chat_gen"
HF_TYPE="chat"
DEBUG_FLAG="--debug"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --hf-type)
            HF_TYPE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --no-debug)
            DEBUG_FLAG=""
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model-path PATH     模型路径 (必需)"
            echo "  --datasets DATASETS   数据集列表，用空格分隔 (默认: demo_gsm8k_chat_gen)"
            echo "  --hf-type TYPE        模型类型: chat 或 base (默认: chat)"
            echo "  --work-dir DIR        输出目录 (默认: outputs/evaluation)"
            echo "  --no-debug            非调试模式"
            echo "  --help                显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 必须指定 --model-path 参数"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 运行评估
echo "开始评估..."
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASETS"
echo "模型类型: $HF_TYPE"
echo "输出目录: $WORK_DIR"
echo ""

python run.py \
    --datasets $DATASETS \
    --hf-type $HF_TYPE \
    --hf-path "$MODEL_PATH" \
    --max-out-len 1024 \
    -w "$WORK_DIR" \
    $DEBUG_FLAG

echo ""
echo "评估完成！结果保存在: $WORK_DIR"

