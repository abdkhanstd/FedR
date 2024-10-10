import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader  # Import DataLoader here
from torchvision import transforms
import cv2
import numpy as np


class CompanyDataset(Dataset):
    def __init__(self, csv_file=None, data_type='train', transform=None, mask_transform=None, img_size=(224, 224), validation=False):
        self.validation = validation
        self.img_size = img_size
        self.transform = transform
        self.mask_transform = mask_transform

        if not validation:
            # Load all data from the CSV file, disregard the 'type' column
            self.data = pd.read_csv(csv_file)
            #self.data = pd.read_csv(csv_file, nrows=500) # This is just for debug

        else:
            # For validation, we load the KITTIRoad testing set
            self.images_dir = 'datasets/KITTIRoad/testing/images'
            self.image_files = os.listdir(self.images_dir)

    def __len__(self):
        if self.validation:
            return len(self.image_files)
        return len(self.data)

    def __getitem__(self, idx):
        if self.validation:
            # Validation case (KITTIRoad dataset)
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_name = img_name.replace('_', '_road_')
            mask_path = os.path.join('datasets/KITTIRoad/testing/masks', mask_name)

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found at path: {img_path}")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask file not found at path: {mask_path}")

        else:
            # Training case (using CSV)
            img_path = os.path.join('datasets/', self.data.iloc[idx]['image'])
            mask_path = os.path.join('datasets/', self.data.iloc[idx]['mask'])


            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found at path: {img_path}")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask file not found at path: {mask_path}")

        # Try loading the image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"OpenCV failed to load image at {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error loading image: {img_path}. Error: {e}")

        # Try loading the mask
        try:
            mask = cv2.imread(mask_path)
            if mask is None:
                raise ValueError(f"OpenCV failed to load mask at {mask_path}")

            # Adjust mask if KITTIRoad or RoadAnomaly datasets are used
            if 'RoadAnomaly' in mask_path:
                mask = np.all(mask == [128, 64, 128], axis=-1).astype(np.uint8) * 255  # Binary mask for RoadAnomaly
            elif 'KITTIRoad' in mask_path:
                mask = np.all(mask == [255, 0, 255], axis=-1).astype(np.uint8) * 255  # Binary mask for KITTIRoad
            else:
                # For other datasets, assuming the mask is grayscale
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask
        except Exception as e:
            raise ValueError(f"Error loading mask: {mask_path}. Error: {e}")

        # Resize image and mask using OpenCV
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def custom_collate_fn(batch):
    # Remove items where image or mask is None
    batch = [item for item in batch if item[0] is not None and item[1] is not None]

    if len(batch) == 0:
        return None, None

    return torch.utils.data.dataloader.default_collate(batch)


def get_company_dataloaders(csv_file, batch_size, img_size=(224, 224), collate_fn=None):
    # Define augmentations for training data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=True),  # Explicitly set antialias to True
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a 50% chance
        transforms.RandomRotation(degrees=15),  # Randomly rotate the image by ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations (scaling & translation)
        transforms.RandomCrop(img_size),  # Random crop to the same size
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Occasionally apply Gaussian blur
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=True),  # Explicitly set antialias to True for validation
    ])

    # Mask transform (resize and convert to tensor)
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
    ])

    # Create datasets for train and validation
    train_dataset = CompanyDataset(csv_file, data_type='train', transform=train_transform, mask_transform=mask_transform, img_size=img_size)
    val_dataset = CompanyDataset(validation=True, transform=val_transform, mask_transform=mask_transform, img_size=img_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader    
