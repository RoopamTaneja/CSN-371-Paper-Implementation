import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import argparse
import os

from conv_vision_transformer_model import CViT
from utils.sessions import session

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, dataloaders, dataset_sizes, device, optimizer, criterion, scheduler, batch_size):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.batch_size = batch_size

    def train(self, num_epochs, min_val_loss=1e4):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        min_loss = min_val_loss

        train_loss, train_accu, val_loss, val_accu = [], [], [], []

        for epoch in range(num_epochs):
            print(f"\n{'='*30}\nEpoch {epoch+1}/{num_epochs}\n{'='*30}")
            for phase in ["train", "validation"]:
                print(f"\n--- Phase: {phase.upper()} ---")
                self.model.train() if phase == "train" else self.model.eval()
                running_loss, running_corrects, phase_idx = 0.0, 0, 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    if phase_idx % 200 == 0:
                        print(f"[{phase.capitalize()} Batch {phase_idx:4d}] Loss: {loss.item():.6f}")
                    phase_idx += 1
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    self.scheduler.step()
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                if phase == "train":
                    train_loss.append(epoch_loss)
                    train_accu.append(epoch_acc.item())
                else:
                    val_loss.append(epoch_loss)
                    val_accu.append(epoch_acc.item())
                print(f"\n{phase.capitalize()} Summary: Loss = {epoch_loss:.4f} | Accuracy = {epoch_acc:.4f}")
                if phase == "validation" and epoch_loss < min_loss:
                    print(f"Validation loss improved: {min_loss:.6f} → {epoch_loss:.6f}. Saving model ...")
                    min_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print(f"{'-'*30}\nEnd of Epoch {epoch+1}\n{'-'*30}")
            print(f"Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_accu[-1]:.4f}")
            print(f"Val   Loss: {val_loss[-1]:.4f} | Val   Acc: {val_accu[-1]:.4f}")

        self.model.load_state_dict(best_model_wts)
        state = {"epoch": num_epochs, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "min_loss": min_loss}
        os.makedirs("saved_model", exist_ok=True)
        torch.save(state, "saved_model/trained_model.pth")
        print("\nTraining complete. Best validation loss: {:.6f}".format(min_loss))
        return train_loss, train_accu, val_loss, val_accu, min_loss

    def test(self):
        self.model.eval()
        correct = 0
        total = self.dataset_sizes["test"]
        for inputs, labels in self.dataloaders["test"]:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs).float()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
        accuracy = (correct / total) * 100
        print(f"\n{'='*30}\nTest Set Accuracy: {accuracy:.2f}%\n{'='*30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--epoch", "-e", type=int, default=1, help="Number of epochs for training (default: 1).")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Training data path.")
    parser.add_argument("--batch", "-b", type=int, default=32, help="Batch size (default: 32).")
    args = parser.parse_args()

    dir_path = args.dir
    batch_size = args.batch
    epoch = args.epoch

    batch_size, dataloaders, dataset_sizes = session(dir_path, batch_size)
    model = CViT(image_size=224, patch_size=7, num_classes=2, cnn_channels=512, transformer_dim=1024, transformer_depth=6, transformer_heads=8, transformer_mlp_dim=2048).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    trainer = Trainer(model, dataloaders, dataset_sizes, device, optimizer, criterion, scheduler, batch_size)
    trainer.train(epoch)
    trainer.test()
