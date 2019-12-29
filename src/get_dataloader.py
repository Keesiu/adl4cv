#!/usr/bin/env python
# coding: utf-8

### General stuff

#### Import libraries

import torch
import numpy as np
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import os

def main(DATA_DIR = os.path.abspath("./data/256_ObjectCategories"),
         TEST_RATIO = 0.1,
         VAL_RATIO = 0.1,
         RS = 42, # Random seed
         BS = 32, # Batch size
         TOTAL_MEANS = [0.485, 0.456, 0.406],  # Imagenet means per RGB-channel, use None to calculate it
         # use [0.5520134568214417, 0.533597469329834, 0.5050241947174072] to make dataset specific
         TOTAL_STDS = [0.229, 0.224, 0.225],  # Imagenet stds per RGB-channel, use None to calculate it
         # use [0.03332509845495224, 0.03334072232246399, 0.0340290367603302] to make dataset specific
         RANDOM_AFFINE = 5,
         RESIZE = (224, 224),
         RANDOM_CROP = (160, 160)
        ):

    #### Set seeds for reproducibility
    np.random.seed(RS)
    torch.manual_seed(RS)

    ### Get normalization parameters from total data

    #### load total data
    total_data = datasets.ImageFolder(DATA_DIR, transforms.ToTensor())
    total_n = len(total_data)
    total_targets = total_data.targets

    #### calculate normalization parameters
    if TOTAL_MEANS == None or TOTAL_STDS == None:
        image_means = torch.stack([tensor.mean(1).mean(1) for tensor, _ in total_data])
        image_stds = torch.stack([tensor.std(1).std(1) for tensor, _ in total_data])
        total_means = image_means.mean(0).tolist()
        total_stds = image_stds.std(0).tolist()
        print("Average means per RGB-channel {}".format(total_means))
        print("Average standard deviation per RGB-channel {}".format(total_stds))
    else:
        total_means = TOTAL_MEANS
        total_stds = TOTAL_STDS

    ### Create dataloaders from transformed datasets

    #### define tranformations
    train_transform = transforms.Compose([transforms.RandomAffine(RANDOM_AFFINE),
                                          transforms.Resize(RESIZE),
                                          transforms.RandomCrop(RANDOM_CROP),
                                          transforms.Resize(RESIZE),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(total_means, total_stds)
                                         ])
    val_transform = transforms.Compose([transforms.Resize(RESIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize(total_means, total_stds)
                                       ])
    test_transform = transforms.Compose([transforms.Resize(RESIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(total_means, total_stds)
                                        ])

    #### load and tranform datasets
    train_data = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    test_data = datasets.ImageFolder(DATA_DIR, transform=test_transform)

    #### define stratified train-val-test split
    train_idx, test_idx = train_test_split(
        np.arange(total_n),
        test_size=int(TEST_RATIO*total_n), random_state=RS, shuffle=True, stratify=total_targets)

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=int(VAL_RATIO*total_n), random_state=RS, shuffle=True, stratify=np.array(total_targets)[train_idx])

    #### define dataloader with different subsets
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=BS)
    val_loader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=BS)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=BS)

    return train_loader, val_loader, test_loader
