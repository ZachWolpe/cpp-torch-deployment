"""
--------------------------------------------------------------------------------
exe_1_download_data.py

Download the training/testing data. Save the data in `.pt` format for easy loading.

: zach.wolpe@medibio.com.au
: 16-09-2024
--------------------------------------------------------------------------------
"""

from torchvision.transforms import ToTensor
from torchvision import datasets
import torch

if __name__ == '__main__':
    print('Downloading training data...')
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    testing_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # save
    output_path = "data"
    torch.save(training_data, f'{output_path}/training_data.pt')
    torch.save(testing_data, f'{output_path}/testing_data.pt')
    print(f'Training & testing data saved to directory: {output_path}.')
    print('Downloading complete.')

    # load data
    training_data = torch.load(f'{output_path}/training_data.pt')
    testing_data = torch.load(f'{output_path}/testing_data.pt')
    assert training_data, 'Training data unable to load.'
    assert testing_data, 'Testing data unable to load.'
    print('Test complete - data is able to load sucessfully. Use <torch.load(PATH.pt)> to load the data.')

    print('Data download and save complete.')