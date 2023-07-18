import torch
import torchvision
from torch import nn

vgg16=torchvision.models.vgg16(pretrained=False)

#保存方式1，模型结构+模型参数
torch.save(vgg16,"vgg16_method1_path")

#保存方式2，模型参数(官方推荐)
torch.save(vgg16.state_dict(),"vgg16_method2_path")

#保存方式1加载模型
model=torch.load("vgg16_method1_path")

#保存方式2加载模型
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict("vgg16_method2_path")
