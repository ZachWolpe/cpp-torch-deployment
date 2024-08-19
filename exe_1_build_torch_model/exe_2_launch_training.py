"""
--------------------------------------------------------------------------------
exe_2_launch_traning.py

Launch a training job. Checkpointing is used to save the model every 500 batches (which allows for resuming training).

: zach.wolpe@medibio.com.au
: 16-09-2024
--------------------------------------------------------------------------------
"""

from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchinfo import summary
from torch import nn
import numpy as np
import warnings
import argparse
import torch
import mlflow
import os

from neural_network_architecture import (NeuralNetwork, train, load_checkpoint, get_epoch_number)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Launching training job...')

    # take args
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('-d', '--data_path', type=str, default='data', help='Path to training/testing data.')
    parser.add_argument('-r', '--resume', type=bool, default=True, help='Resume training from checkpoint.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')

    # load args
    args = parser.parse_args()
    DATA_PATH = args.data_path
    EPOCHS = args.epochs
    RESUME = args.resume
    BATCH_SIZE = args.batch_size

    # compute the epoch number
    if RESUME:
        checkpoint_files = os.listdir('checkpoints')
        CHECKPOINT_PATH, LAST_EPOCH = get_epoch_number(checkpoint_files)
    else:
        CHECKPOINT_PATH, LAST_EPOCH = None, 0
        # delete files in checkpoint directory
        for f in os.listdir('checkpoints'):
            os.remove(f'./checkpoints/{f}')

    # setup runtime
    END_EPOCH = LAST_EPOCH + EPOCHS + 1
    NEW_RUN = True if LAST_EPOCH == 0 else False
    START_EPOCH = LAST_EPOCH + 1

    # Get cpu or gpu for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # log runtime
    msg = f"""
    --+-- Training a neural network --+--
    {parser.description}

        -> relaunching training: {RESUME}
        -> start epoch: {START_EPOCH}
        -> end epoch: {END_EPOCH}
        -> device: {device}

    --+--+--+--+--+--+--+--+--+--+--+--+-
    """
    print(msg)

    # load data ------------------------------------------------------------------------------++
    training_data = torch.load(f'{DATA_PATH}/training_data.pt')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # load data ------------------------------------------------------------------------------++

    # define model architecture --------------------------------------------------------------++
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # define model architecture --------------------------------------------------------------++

    # mlflow model signature --- -------------------------------------------------------------++
    input = np.random.uniform(size=[1, 28, 28])
    input = torch.tensor(input).float()
    signature = mlflow.models.infer_signature(
        input.numpy(),
        model(input).detach().numpy(),
        )
    # print('signature: ', signature)
    # mlflow model signature -----------------------------------------------------------------++

    # test if mlflow is running, if so end the run
    if mlflow.active_run():
        mlflow.end_run()

    # launch traning
    with mlflow.start_run() as run:
        params = {
            "epochs": EPOCHS,
            "start_epoch": START_EPOCH,
            "end_epoch": END_EPOCH,
            "device": device,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "SGD",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not NEW_RUN:
            model, optimizer, start_epoch, start_batch, loss = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        else:
            start_epoch, start_batch = 0, 0

        # Then start your training loop from start_epoch and start_batch
        for t in range(START_EPOCH, END_EPOCH):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, device, model, loss_fn, metric_fn, optimizer, t, checkpoint_dir)
    
        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model", signature=signature)

    # Register the model with the latest run -------------------------------------------------++
    client = MlflowClient()
    run_id = run.info.run_id
    model_name = "fashion_mnist_classifier"
    model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)

    print(f"Model registered with name: {model_name}")
    print(f"Model version: {model_version.version}")
    # Register the model with the latest run -------------------------------------------------++

    print('Training complete. Runtime complete.')