"""
--------------------------------------------------------------------------------
exe_4_convert_model_to_torchscript.py

Before loading the model in C++, we need to convert the model to TorchScript. This removes the need for the Python interpreter when loading the model in C++.

Pass the desired model path to `exe_3_save_model.py` as a command line argument.

Only loading with checkpoint paths is demonstrated here. See `exe_3_perform_inference.py` for loading with MLflow run_id and model registry.

[TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

: zach.wolpe@medibio.com.au
: 17-09-2024
--------------------------------------------------------------------------------
"""

from neural_network_architecture import (NeuralNetwork, load_checkpoint, get_epoch_number)
from torchmetrics import Accuracy
from torch import nn
import argparse
import torch
import os

# default model path
DEFAULT_MODEL_PATH = './checkpoints/checkpoint_epoch_1_batch_500.pth'
parser = argparse.ArgumentParser(description='Convert model to TorchScript.')
parser.add_argument('-m', '--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to model.')


if __name__ == '__main__':
    # load model path
    args = parser.parse_args()
    MODEL_PATH = args.model_path

    # load model
    state_dict = torch.load(MODEL_PATH)

    # define model architecture --------------------------------------------------------------++
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # define model architecture --------------------------------------------------------------++

    # load model weights with checkpoint -----------------------------------------------------++
    checkpoint_files = os.listdir('checkpoints')
    CHECKPOINT_PATH, LAST_EPOCH = get_epoch_number(checkpoint_files)
    model, optimizer, start_epoch, start_batch, loss = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    # load model weights with checkpoint -----------------------------------------------------++

    # Set the model to evaluation mode
    model.eval()

    # save the model with TorchScript
    MODEL_NAME = 'model_scripted.pt'
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(MODEL_NAME) # Save
    print('Model converted and saved as model_scripted.pt')

    # load: test
    model = torch.jit.load(MODEL_NAME)
    print('Model loaded: ', model)

    print("Model converted to TorchScript and saved as model_scripted.pt")