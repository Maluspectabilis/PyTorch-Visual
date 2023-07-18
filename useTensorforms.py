from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img=Image.open("dataset/val/bees/144098310_a4176fd54d.jpg")

writer=SummaryWriter("log")

#ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize归一化
trans_norm=transforms.Normalize([1,3,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
writer.add_image("Normalize",img_norm)

#Resize缩放
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)#img PIL  -->  resize   -->  img_resize PIL
img_resize=trans_totensor(img_resize)# img_resize PIL -->  tensor   -->  img_resize tensor
writer.add_image("Resize",img_resize,0)

#Compose等比缩放，不改变比例
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop随机裁剪
trans_random=transforms.RandomCrop(200)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RamdomCrop",img_crop,i)

writer.close()