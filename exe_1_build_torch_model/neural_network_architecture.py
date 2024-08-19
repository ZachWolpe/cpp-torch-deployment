"""
--------------------------------------------------------------------------------
neural_network_architecture.py

Define the model and helper functions used in it's execution.

: zach.wolpe@medibio.com.au
: 16-09-2024
--------------------------------------------------------------------------------
"""

from torch import nn
import mlflow
import torch
import os


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(train_dataloader, device, model, loss_fn, metrics_fn, optimizer, epoch, checkpoint_dir):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric("loss", loss, step=(epoch * len(train_dataloader) + batch))
            mlflow.log_metric("accuracy", accuracy.item(), step=(epoch * len(train_dataloader) + batch))
            print(
                f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(train_dataloader)}]"
            )

        # Save checkpoint every 500 batches
        if batch % 500 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch': batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch}.pth')
            torch.save(checkpoint, checkpoint_path)
            mlflow.log_artifact(checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch = checkpoint['batch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, batch, loss


def get_epoch_number(checkpoint_files):
    if checkpoint_files:
        LAST_EPOCH = max([int(f.split('_')[2]) for f in checkpoint_files])
        CHECKPOINT_PATH = f'./checkpoints/checkpoint_epoch_{LAST_EPOCH}_batch_500.pth'
    else:
        print('No checkpoint files found. Starting training from epoch 0.')
        LAST_EPOCH = 0
        CHECKPOINT_PATH = None
    return CHECKPOINT_PATH, LAST_EPOCH
