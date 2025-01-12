import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric


# Define a Custom Accuracy Metric
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)  # Get predicted class
        assert preds.shape == target.shape  # Ensure shapes match
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


# Define the Lightning Module
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten the input
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)

        # Compute metrics
        acc = self.accuracy(scores, y)
        custom_acc = self.my_accuracy(scores, y)
        f1 = self.f1_score(scores, y)

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True)
        self.log(f"{stage}_my_accuracy", custom_acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Hyperparameters
input_size = 784  # MNIST images are 28x28
num_classes = 10  # Digits 0-9
batch_size = 64
num_epochs = 3

# Load Data
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

# Initialize Model
model = NN(input_size=input_size, num_classes=num_classes)

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=num_epochs,
    precision=16,
)

# Training and Validation
trainer.fit(model, train_loader, val_loader)

# Testing
trainer.test(model, test_loader)
