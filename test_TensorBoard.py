from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer= SummaryWriter("log")

image_path="dataset/val/bees/59798110_2b6a3c8031.jpg"
image_PIL=Image.open(image_path)
image_array=np.array(image_PIL)#转换为numpy类型

writer.add_image("test",image_array,1,dataformats='HWC')#1,表示第一步
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()