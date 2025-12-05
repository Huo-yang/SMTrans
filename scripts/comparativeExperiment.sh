source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_hy

# 切换到脚本文件所在目录的上一级目录
cd "$(dirname "$0")/.."

# 定义模型名称变量
model_name="smt_128"
times=20

# 循环执行训练
for ((i=1; i<=times; i++)); do
    echo "==== Running train.py: Iteration $i ===="
    python train.py --model "${model_name}" --dataset_name 'RML2016.10a' --result_root_path "./results/10a/${model_name}"
    # python train.py --model "${model_name}" --dataset_name 'RML2016.10b' --result_root_path "./results/10b/${model_name}"
    # python train.py --model "${model_name}" --dataset_name 'RML2018.01a' --result_root_path "./results/01a/${model_name}"
done
