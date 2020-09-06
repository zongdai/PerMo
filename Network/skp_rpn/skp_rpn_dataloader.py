from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import torch
import math
class Kp_Range_Dataset(Dataset):

    def __init__(self, structrue_dir, label_dir , min_x=-25, max_x=25, min_z=0, max_z=50, step=8, fg_dis=5, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])):
        self.structrue_dir = structrue_dir
        self.label_dir = label_dir
        self.labels = {}
        self.anchor_vector = {}
        self.transform = transform
        xs = []
        zs = []
        for f in os.listdir(label_dir):
            with open(os.path.join(label_dir, f)) as ff:
                line = ff.readline()
                [x, y, z, a, b, c, width, height, length] = [float(x) for x in line.split()]
                    
                if x > min_x and x < max_x and z > min_z and z < max_z:
                    anchor_vec = np.zeros((1, ((max_x-min_x)*(max_z-min_z))//(step*step)))
                    c = 0
                    for x_i in range(min_x, max_x, step):
                        for z_i in range(min_z, max_z, step):
                            if math.sqrt((x_i-x)*(x_i-x) + (z_i-z)*(z_i-z)) < fg_dis:
                                anchor_vec[0][c] = 1
                            c += 1
                    self.labels[f.split('.')[0]] = [x, y, z, b, width-1.5, height-1.5, length-4]
                    self.anchor_vector[f.split('.')[0]] = anchor_vec
                    xs.append(x)
                    zs.append(z)
        xs = np.array(xs) 
        zs = np.array(zs)
        self.x_mean = np.mean(xs)
        self.x_std = np.std(xs)
        self.z_mean = np.mean(zs)
        self.z_std = np.std(zs)      
        self.kp_items = [f for f in os.listdir(structrue_dir) if f.split('.')[0] in self.labels]
    def __len__(self):
        return len(self.kp_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        kp = np.load(os.path.join(self.structrue_dir, self.kp_items[idx]))

        # kp_name = self.kp_items[idx].split('_')[0]
        # kp_index = int((self.kp_items[idx].split('_')[1]).split('.')[0])
        # print(kp_name)
        # print(x,y,z)
        [x, y, z, b, width, height, length] = [torch.tensor(k) for k in self.labels[self.kp_items[idx].split('.')[0]]]
        anchor_vec = self.anchor_vector[self.kp_items[idx].split('.')[0]]
        # x = (x - self.x_mean)/self.x_std * 10
        # z = (z - self.z_mean)/self.z_std * 10
        anchor_vec = anchor_vec.astype('float32')
        kp = kp.astype('float32')
        if self.transform:
            kp = self.transform(kp)
            anchor_vec = self.transform(anchor_vec)
        
        positions = torch.tensor([x,y,z,b])
        sizes = torch.tensor([width, height, length])
        anchor_vecs = anchor_vec
        return kp, positions, sizes, anchor_vecs