import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                # TODO: A little data augmentation FLIPS?!
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),  #Cela ajoutera des retournements horizontaux et verticaux aléatoires aux images pendant l'entraînement, 
                                                  #ce qui peut aider à améliorer la généralisation du modèle.
                transforms.RandomRotation(degrees=60), #"degrees" pour contrôler l'amplitude de la rotation.
           
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # TODO: apply transforms 
        if self.transforms is not None: #Apply transforms if available
          img = self.transforms(img)

        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        # TODO: convert to tensor 

        ##Normalisation d'abord des valeurs L et B
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        ##Conversion
        L_tensor = torch.tensor(L)
        ab_tensor = torch.tensor(ab)
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=1, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader