# callbacks.py

class CustomCallback:
    def on_epoch_start(self, epoch):
        print(f"Epoch {epoch + 1} started.")

    def on_epoch_end(self, epoch, train_loss, val_loss, val_accuracy):
        print(
            f"Epoch {epoch + 1} finished. Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    def on_training_start(self):
        print("Training started.")

    def on_training_end(self):
        print("Training finished.")
