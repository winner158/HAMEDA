import torch
import os

# 查看CUDA是否可用
if torch.cuda.is_available():
    # 获取CUDA设备数量
    device_count = torch.cuda.device_count()
    print(f'CUDA device count: {device_count}')

    # 遍历每个设备并打印设备编号
    for i in range(device_count):
        print(f'Device {i} name: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA is not available.')


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '0:large_pool,30GB'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '0:large_pool,30GB;1:large_pool,30GB'

# 获取当前的PYTORCH_CUDA_ALLOC_CONF值
cuda_alloc_conf = os.getenv('PYTORCH_CUDA_ALLOC_CONF')
# 直接修改环境变量即可，建议在 Python 运行过程中临时修改，避免不必要的性能降低
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:10240'

print(f"PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")