from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch
from torchvision import transforms

from models.xresnet_hybrid import xresnet_hybrid101
from utils.utils import standardize, inv_standardize, file_path,dir_path
from utils.custom_activation_functions import Mish_layer
from utils.custom_loss_functions import root_mean_squared_error, mae_loss_wgtd
from data.custom_datasets import RegressionNumpyArrayDataset

import argparse
import numpy as np


def main(path_to_images,path_to_labels,mmap_mode,path_to_weights,output_dir):
    # Path to the dataset
    path_to_images = path_to_images
    path_to_masses = path_to_labels
    # Load the dataset
    images = np.load(path_to_images,mmap_mode='r' if mmap_mode else None).astype('float32')
    images = images.reshape(-1,1,150,150)
    labels = np.load(path_to_masses,mmap_mode='r' if mmap_mode else None).astype('float32')
    labels = labels.reshape(-1,1)
    # Calculate the stats of the dataset to standardize it
    IMAGES_MEAN, IMAGES_STD = images.mean(), images.std()
    LABELS_MEAN, LABELS_STD = labels.mean(), labels.std()

    images = standardize(images,IMAGES_STD,IMAGES_MEAN)
    labels = standardize(labels,LABELS_STD,LABELS_MEAN)
    # Split the dataset into train, valid, test subdatasets
    np.random.seed(234)
    indexes = list(range(labels.shape[0]))
    # Define transforms
    base_image_transforms = [
        transforms.Resize(150)
    ]
    # Create the test dataset
    test_dataset = RegressionNumpyArrayDataset(images, labels, indexes, transforms.Compose(base_image_transforms))
    # Create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create model
    torch.manual_seed(50)
    # N_out is the number of output neurons in the last linear layer.
    # C_in is the number of channels in the input images.
    model = xresnet_hybrid101(n_out=1, sa=True, act_cls=Mish_layer, c_in=1,device=device)
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()
    # Create a test dataloader
    #test_dl = DataLoader(test_dataset, batch_size=1,shuffle=False,device=device)
    #with torch.no_grad():
    # TODO finish
    #m_pred,m_true = learn.get_preds(dl=test_dl,reorder=False)
    # Unstandartize the dataset
    #m_pred,m_true = inv_standardize(m_pred),inv_standardize(m_true)
    # Save the predicted and true values in an .npz file
    #full_path = os.path.join(output_dir,'resnet101_predicted_density_masses')
    #np.savez(full_path,m_pred=m_pred,m_true=m_true)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict density masses for the dataset by using trained resnet101.')

    parser.add_argument('--path_to_images', required=True,type=file_path,
                        help='The path to a .npy file with images. It has to have the following dimensions: (num_of_elements,1,150,150).')
    parser.add_argument('--path_to_labels', required=True,type=file_path,
                    help='The path to a .npy file with density masses. It has to have the following dimensions: (num_of_elements,1).')
    parser.add_argument('--path_to_weights', required=True,type=file_path,
        help='The path to a .npy file with density masses. It has to have the following dimensions: (num_of_elements,1).')
    parser.add_argument('--output_dir', required=True,type=dir_path,
                help='The directory where the best_model.pth (weights of the model) file will be stored after training.')


    args = parser.parse_args()

    main(path_to_images=args.path_to_images,
        path_to_labels=args.path_to_labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        mmap_mode=args.mmap_mode,
        num_of_epochs=args.num_of_epochs,
        lr=args.lr)