# train.py

import torch
from tqdm import tqdm
from dataset import get_custom_data_loaders
from model import DeepNN
from metrics import CustomAccuracy
from callbacks import CustomCallback
import config
from torch.profiler import profile, ProfilerActivity, record_function

def train_epoch(model, loader, optimizer, criterion, device, metric, profiler=None):
    model.train()
    metric.reset()
    epoch_loss = 0
    for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Training")):
        data, targets = data.to(device), targets.to(device)
        data = data.view(data.size(0), -1)

        # Forward pass
        with record_function("forward_pass"):
            scores = model(data)

        # Compute loss
        with record_function("compute_loss"):
            loss = criterion(scores, targets)

        # Backward pass
        with record_function("backward_pass"):
            optimizer.zero_grad()
            loss.backward()

        # Optimizer step
        with record_function("optimizer_step"):
            optimizer.step()

        # Update metric
        metric.update(scores, targets)
        epoch_loss += loss.item()

        # Profiler step
        if profiler:
            profiler.step()

    accuracy = metric.compute()
    return epoch_loss / len(loader), accuracy

def evaluate_model(model, loader, criterion, device, metric, profiler=None):
    model.eval()
    metric.reset()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Validation")):
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.size(0), -1)

            # Forward pass
            with record_function("forward_pass"):
                scores = model(data)

            # Compute loss
            with record_function("compute_loss"):
                loss = criterion(scores, targets)

            # Update metric
            metric.update(scores, targets)
            val_loss += loss.item()

            # Profiler step
            if profiler:
                profiler.step()

    accuracy = metric.compute()
    return val_loss / len(loader), accuracy

if __name__ == "__main__":
    # Initialize loaders, model, metric, callback
    train_loader, val_loader, test_loader = get_custom_data_loaders(
        config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS
    )
    model = DeepNN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    metric = CustomAccuracy()
    callback = CustomCallback()

    # Setup PyTorch Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as profiler:
        # Training loop
        callback.on_training_start()
        for epoch in range(config.NUM_EPOCHS):
            callback.on_epoch_start(epoch)

            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, criterion, config.DEVICE, metric, profiler
            )
            val_loss, val_accuracy = evaluate_model(
                model, val_loader, criterion, config.DEVICE, metric, profiler
            )

            callback.on_epoch_end(epoch, train_loss, val_loss, val_accuracy)

        callback.on_training_end()

    # Output profiler results
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Test model
    test_loss, test_accuracy = evaluate_model(
        model, test_loader, criterion, config.DEVICE, metric
    )
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
