import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

# Define the data categories (Training or testing)
data_cat = ['train', 'valid']

# Create Dataframes containing study path, image count per study, and label
def get_study_data(study_type):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}

    for category in data_cat:
        base_directory = 'D:\Desktop\FYP\MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(base_directory))[0][1]
        study_data[category] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0  # Used to index the rows of the data frame (table)

        #Patient can have multiple studies, include path, count and label for every study in df
        for patient in tqdm(patients):
            for study in os.listdir(base_directory + patient):
                path = base_directory + patient + '/' + study + '/'
                label = study_label[study.split('_')[1]]
                study_data[category].loc[i] = [path, len(os.listdir(path)), label]
                i += 1
    return study_data

# ImageDataset class
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform  # Transforms include rotations / scaling / etc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        # Load and stack images in same study together
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i + 1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        # Create sample with images and label
        sample = {'images': images, 'label': label}

        return sample

# Custom collate function for dataloaders
def my_collate(batch):
    data = [item['images'] for item in batch]  # form a list of tensor
    target = [item['label'] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

# Returns dataloader pipeline with data augmentation
def get_dataloaders(data, batch_size):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create and transform samples
    image_datasets = {x: ImageDataset(data[x], transform = data_transforms[x]) for x in data_cat}
    #Batch samples together
    dataloaders = {x: DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True, num_workers = 8, collate_fn = my_collate) for x in data_cat}

    return dataloaders


