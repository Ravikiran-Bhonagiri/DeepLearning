import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


# Define the PyTorch Lightning model
class LightningNN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
num_classes = 10  # Digits 0-9
batch_size = 64
num_epochs = 5

# Data preparation
transform = transforms.ToTensor()
entire_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transform, download=True
)
train_ds, val_ds = random_split(entire_dataset, [50000, 10000])
test_ds = datasets.MNIST(
    root="dataset/", train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

# Initialize the PyTorch Lightning model
model = LightningNN(input_size=input_size, num_classes=num_classes)

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)
