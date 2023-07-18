from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path="dataset/val/bees/144098310_a4176fd54d.jpg"
img=Image.open(img_path)

writer = SummaryWriter("log")

tensor_trans=transforms.ToTensor()#返回一个totensor的对象
tensor_image=tensor_trans(img)#将图片转换为tensor的image

writer.add_image("tensor_img",tensor_image)

writer.close()