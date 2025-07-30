# _*_ coding: utf-8 _*_
'''
时间:      2025/7/30 15:19
@author:  andinm
'''
import torch
print(f"hello pytorch {torch.__version__}")
print(f"cuda is avaliable: {torch.cuda.is_available()}")
'''hello pytorch 2.5.1
cuda is avaliable: True'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")
