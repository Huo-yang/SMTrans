# 让 conda 命令在 PowerShell 脚本中可用
& "D:\miniconda3\shell\condabin\conda-hook.ps1"
conda activate pytorch_hy

# 切换到脚本文件所在目录的上一级目录
Set-Location (Resolve-Path "$PSScriptRoot\..").Path

# 定义模型名称变量
$model_name = "MCLDNN"
$times = 10

# 循环执行训练
for ($i = 1; $i -le $times; $i++) {
    Write-Host "==== Running train.py: Iteration $i ===="
    python train.py --model $model_name --dataset_name 'RML2016.10a' --result_root_path "./results/10a/$model_name"
    # python train.py --model $model_name --dataset_name 'RML2018.01a' --result_root_path "./results/01a/$model_name"
}