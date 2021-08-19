from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch
from torchvision import transforms

from models.xresnet_hybrid import xresnet_hybrid101
from utils.utils import standardize, inv_standardize
from utils.custom_activation_functions import Mish_layer
from utils.custom_loss_functions import root_mean_squared_error, mae_loss_wgtd
from data.custom_datasets import RegressionNumpyArrayDataset

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings

matplotlib.use('Agg')
#%matplotlib inline
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # Path to the dataset
    path_to_images = r'D:\Education\GSoC\DeepLense\coding\mass_density_lenses_generated\orign_gener_vort_25k_images.npy'
    path_to_masses = r'D:\Education\GSoC\DeepLense\coding\mass_density_lenses_generated\orign_gener_vort_25k_masses.npy'
    # Load the dataset
    images = np.load(path_to_images,mmap_mode='r')[:100].astype('float32')
    images = images.reshape(-1,1,150,150)
    labels = np.load(path_to_masses,mmap_mode='r')[:100].astype('float32')
    labels = labels.reshape(-1,1)
    # Calculate the stats of the dataset to standardize it
    IMAGES_MEAN, IMAGES_STD = images.mean(), images.std()
    LABELS_MEAN, LABELS_STD = labels.mean(), labels.std()

    images = standardize(images,IMAGES_STD,IMAGES_MEAN)
    labels = standardize(labels,LABELS_STD,LABELS_MEAN)
    # Split the dataset into train, valid, test subdatasets
    np.random.seed(234)
    num_of_images = labels.shape[0]
    # 85% for train
    # 10% for valid
    # 5% for test
    max_indx_of_train_images = int(num_of_images*0.85)
    max_indx_of_valid_images = max_indx_of_train_images + int(num_of_images*0.1)
    max_indx_num_of_test_images = max_indx_of_valid_images + int(num_of_images*0.05)
    permutated_indx = np.random.permutation(num_of_images)
    train_indx = permutated_indx[:max_indx_of_train_images]
    valid_indx = permutated_indx[max_indx_of_train_images:max_indx_of_valid_images]
    test_indx = permutated_indx[max_indx_of_valid_images:]
    # Define transforms
    base_image_transforms = [
        transforms.Resize(150)
    ]
    rotation_image_transofrms = [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0,360))
    ]
    # Crete datasets
    train_dataset = RegressionNumpyArrayDataset(images, labels, train_indx, transforms.Compose(base_image_transforms+rotation_image_transofrms))
    valid_dataset = RegressionNumpyArrayDataset(images, labels, valid_indx, transforms.Compose(base_image_transforms))
    test_dataset = RegressionNumpyArrayDataset(images, labels, test_indx, transforms.Compose(base_image_transforms))
    # Create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    dls = DataLoaders.from_dsets(train_dataset,valid_dataset,batch_size=batch_size, device=device, num_workers=2)
    # Create model
    torch.manual_seed(50)
    model = xresnet_hybrid101(n_out=1, sa=True, act_cls=Mish_layer, c_in=1,device=device)
    # Create learner
    learn = Learner(
        dls, 
        model,
        opt_func=ranger, 
        loss_func= root_mean_squared_error,  
        metrics=[mae_loss_wgtd],
        model_dir = ''
    )
    # Find lr
    # learn.lr_find()
    # Train the model
    learn.fit_one_cycle(120,1e-2,cbs=[
       SaveModelCallback(monitor='mae_loss_wgtd',fname='best_model')])