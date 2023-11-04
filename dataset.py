from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, N, images, masks = None, train = True):
        self.images = images
        self.length = N
        self.images = self.images.reshape(self.length, 64, 64, 3)
        self.masks = masks
        self.train = train
        if self.train:
            self.masks = self.masks.reshape(self.length, 64, 64)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        batch = None
        if self.train:
            mask = self.masks[idx]
            mask = torch.tensor(mask, dtype=torch.long)
            batch = {
                'image': image,
                'mask': mask
            }
        else:
            batch = {
                'image': image
            }
        return batch