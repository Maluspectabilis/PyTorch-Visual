from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir="dataset/train"
ants_label_dir="ants"
ants_dataset=MyData(root_dir,ants_label_dir)#蚂蚁的数据集
bees_label_dir="ants"
bees_dataset=MyData(root_dir,bees_label_dir)#蜜蜂的数据集

train_dataset=ants_dataset+bees_dataset#两个数据集相加，整个数据集

img,label=ants_dataset[1]
img.show()