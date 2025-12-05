import torch
import time
from thop import profile

def test_model_performance(model, input_data, warmup=10, runs=400):
    # 1. 初始化模型并迁移到GPU
    model = model.cuda()
    model.eval()  # 评估模式

    # 2. 构造输入数据
    if isinstance(input_data, tuple):
        input_data = tuple(x.cuda() for x in input_data)
    else:
        input_data = input_data.cuda()

    # 3. 预热 GPU，避免首次运行偏慢
    for _ in range(warmup):
        _ = model(input_data)

    # 4. 计算平均推理时间
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        _ = model(input_data)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / runs
    print(f"Average inference time: {avg_time*1000:.3f} ms")

    # 5. 统计 FLOPs 和参数量
    macs, params = profile(model, inputs=(input_data,), verbose=False)
    print(f"FLOPs: {macs / 1e6:.2f} M")
    print(f"Params: {params / 1e3:.2f} K")