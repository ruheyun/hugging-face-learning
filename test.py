import torch
import transformers

print("检测系统中是否有可用的GPU:", torch.cuda.is_available())

if torch.cuda.is_available():
    
    print(f"GPU可用，可用的GPU设备数量：{torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU设备{i}: {torch.cuda.get_device_name(i)}")
    
print(f'transformers 的版本为：{transformers.__version__}')
