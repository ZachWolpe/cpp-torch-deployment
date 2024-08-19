"""
--------------------------------------------------------------------------------
exe_3_perform_inference.py

Load the model and perform inference

: zach.wolpe@medibio.com.au
: 16-09-2024
--------------------------------------------------------------------------------
"""

from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchinfo import summary
from torch import nn
import mlflow

from mlflow.tracking import MlflowClient
import warnings
import torch
import os

from neural_network_architecture import (NeuralNetwork, load_checkpoint, get_epoch_number)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Calling inference...')

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
    print('checkpoint model:')
    print(model)
    # load model weights with checkpoint -----------------------------------------------------++

    # Alternatively,load model with mlflow run_id --------------------------------------------++
    # using run_id
    # run_id = "5d5bed7288604e828f8e7a633dcfbaf5"
    # run_id = "your_mlflow_run_id_here"
    # model_uri = f"runs:/{run_id}/model"
    # # Load the model from MLflow
    # loaded_model = mlflow.pytorch.load_model(model_uri)
    # model = loaded_model.to(device)
    # Alternatively,load model with mlflow run_id --------------------------------------------++

    # using model registry
    # load model from MLflow Model Registry --------------------------------------------------++
    model_name = "fashion_mnist_classifier"
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    model = loaded_model.to(device)

    print('Model loaded from MLflow Model Registry:')
    print(f'Model Name: {model_name}, Version: {latest_version}')
    print(summary(model))
    # Alternatively, load model with mlflow --------------------------------------------------++


    # load data ------------------------------------------------------------------------------++
    DATA_PATH = 'data'
    testing_data = torch.load(f'{DATA_PATH}/testing_data.pt')
    test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

    # testing loop --------------------------------------------------------------------------++
    model.eval()
    total_accuracy = 0
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        accuracy = metric_fn(pred, y)
        total_accuracy += accuracy.item()
        print(f'Batch {batch} accuracy: {accuracy.item()}')
    print(f'Total accuracy (over all batches): {total_accuracy / len(test_dataloader)}')

    # save as .pt for c++
    # model.save("traced_resnet_model.pt")
    print('Inference example complete.')