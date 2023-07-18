import torchvision
from torch import nn

vgg16_false=torchvision.models.vgg16(pretrained=False)#pretrained=False：vgg16使用的网络中的参数是没有经过训练的，初始化的参数
vgg16_true=torchvision.models.vgg16(pretrained=True)

train_data=torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)

print(vgg16_true)
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)




