source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_hy

# 切换到脚本文件所在目录的上一级目录
cd "$(dirname "$0")/.."

# 定义模型名称变量
dataset_name='RML2016.10a'
model_name="smt_128_naive"
times=20
gpu=1

# 循环执行训练
for ((i=1; i<=times; i++)); do
    echo "==== Running train.py: Iteration $i ===="
    python train.py --model "${model_name}" --dataset_name "${dataset_name}" --result_root_path "./results/AE/${dataset_name}/${model_name}" --gpu "${gpu}"
done
