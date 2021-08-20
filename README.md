[![](https://img.shields.io/badge/license-MIT-green)](https://github.com/Claussss/DeepLense-Regression/blob/main/LICENSE.txt)

# DeepLense Regression

A FastAI-based tool for performing regression on strong lensing images to predict axion mass density of galaxies.

This is a Google Summer of Code (GSoC) 2021 project.

## Project Description
The goal of the project is to apply deep regression techniques to explore properties of dark matter. We approximated the mass density of vortex substructure of dark matter condensates on simulated strong lensing images. The images are generated with PyAutoLense.

You can find more information about the project in this [blog post](httos://myBlogpost).

## Installation

Clone the repository and enter the main folder.

```bash
git clone https://github.com/Claussss/DeepLense-Regression.git
cd DeepLense-Regression
```
Next,you need to intall **pipenv** unless you already have it installed
```bush
pip install pipenv
```
Install all the dependencies for the project
```bash
pipenv install --dev
```
Activate the project's virtualenv
```bash
pipenv shell
```
## Usage

### Data
The dataset consists of strong lensing images with vortex substructure generated by PyAutoLense. The parameters for the generator were taken from the following paper: [Decoding Dark Matter Substructure without Supervision](https://arxiv.org/abs/2008.12731).

The dataset contains *25k* grayscale images with the size of *150x150*. 

Images are stored in a .npy file with the following dimensions: *(250000,1,150,150)*. 

Labels are stored in a separate numpy file and have the following dimensions: *(250000,1)*.

You can find the dataset [here](https://drive.google.com/drive/folders/1NPf7Mui5Qt_vVm5wtlNq17vKnto-Nqkc?usp=sharing).
### Weights
We trained xResnetHybrid101 on the provided dataset with the following parameters:
* batch size=64
* number_of_epochs=120
* lr=1e-2

The weights can be found [here](https://drive.google.com/drive/folders/1NPf7Mui5Qt_vVm5wtlNq17vKnto-Nqkc?usp=sharing).
### Examples
[Example notebook](https://github.com/Claussss/DeepLense-Regression/blob/main/example_notebook.ipynb)

### Training
Run the following script with parameters:
```bash
python train.py --path_to_images '/images.npy' 
                --path_to_labels '/labels.npy'
                --output_dir 'output_dir/'
                --batch_size 64
                --num_of_epochs 120
                --lr 1e-2       
```
You can optionally add *--mmap_mode* flag.

| Argument | Description |
| :---         |     :---:      |
| path_to_images | The path to a .npy file with images. It has to have the following dimensions: (num_of_elements,1,150,150). |
| path_to_labels | The path to a .npy file with density masses. It has to have the following dimensions: (num_of_elements,1). |
| output_dir | The directory where the best_model.pth (weights of the model) file will be stored after training. |
| batch_size | Batch siz |
| num_of_epochs | Number of epochs |
| lr | Learning rate |
| mmap_mode | Use the flag if you cannot fit the whole dataset in the RAM. |

### Interference
To run interference on custom strong lensing images, run the following commands:
```bash
python interference.py --path_to_images '/images.npy' 
                       --path_to_weights '/weights.pth'
                       --output_dir 'output_dir/'       
```
You can optionally add *--mmap_mode* flag.

| Argument | Description |
| :---         |     :---:      |
| path_to_images | The path to a .npy file with images. It has to have the following dimensions: (num_of_elements,1,150,150). |
| path_to_weights | The path to a .pth file with trained weights for XResnetHybrid101. |
| output_dir | The directory where the model will output predicted mass densities in a .npy file. |
| mmap_mode | Use the flag if you cannot fit the whole dataset in the RAM. |
