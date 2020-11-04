import cv2
from torch.utils.data import Dataset
import numpy as np

class Data(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    def __getitem__(self, index):
        x, y = self.data.iloc[index]['img'], self.data.iloc[index]['label']
        x = cv2.imread(x)
        if self.transforms is not None:
            res = self.transforms(image=x)
            x = res['image'].astype(np.float32)
        x = x.transpose(2, 0, 1)
        return x, y
    def __len__(self):
        return len(self.data)