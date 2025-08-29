import os
import json
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class LossLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(float(val_loss))


def save_loss(loss_logger, output_dir):
    losses = {'train_losses': loss_logger.train_losses, 'val_losses': loss_logger.val_losses}
    with open(os.path.join(output_dir, 'losses.json'), 'w') as f:
        json.dump(losses, f)


def plot_loss(loss_logger, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_logger.train_losses) + 1)
    plt.plot(epochs, loss_logger.train_losses, 'b-', label='Train Loss')
    if loss_logger.val_losses:
        val_epochs = range(1, len(loss_logger.val_losses) + 1)
        plt.plot(val_epochs, loss_logger.val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()