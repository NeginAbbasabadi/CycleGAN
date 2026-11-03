import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnpairedArtDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.transform = transform
        self.dir_A = os.path.join(root_dir, f'{mode}A') 
        self.dir_B = os.path.join(root_dir, f'{mode}B') 

        self.files_A = sorted(glob.glob(os.path.join(self.dir_A, '*.jpg')))
        self.files_B = sorted(glob.glob(os.path.join(self.dir_B, '*.jpg')))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A_path = self.files_A[index % len(self.files_A)]
        img_B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]

        item_A = Image.open(img_A_path).convert('RGB')
        item_B = Image.open(img_B_path).convert('RGB')

        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return {'A': item_A, 'B': item_B}

art_transforms = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
