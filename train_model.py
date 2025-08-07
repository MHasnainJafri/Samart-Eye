import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix
from PIL import Image

# -------------------------------#
#  Logger Configuration
# -------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CarModelClassifier:
    def __init__(self, data_dir, model_name='resnet50', batch_size=8, num_epochs=100, lr=3e-5):
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.output_dir = None

    def setup_data_loaders(self):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(f"Dataset directories not found: {train_dir}, {val_dir}")

        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

        if len(train_dataset.classes) == 0:
            raise ValueError("No classes found in training dataset!")

        self.class_names = train_dataset.classes
        logger.info(f"Classes: {self.class_names}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=os.cpu_count() if os.name != 'nt' else 0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=os.cpu_count() if os.name != 'nt' else 0, pin_memory=True)

    def setup_model(self):
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        logger.info(f"Loaded pretrained model: {self.model_name}")

        for param in self.model.parameters():
            param.requires_grad = False

        for layer in [self.model.layer2, self.model.layer3, self.model.layer4]:
            for param in layer.parameters():
                param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(self.class_names))
        )

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=7)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Step {batch_idx}, Loss: {loss.item():.4f}")

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.class_names))))
        logger.info(f"Confusion Matrix:\n{cm}")
        return total_loss / len(self.val_loader), 100 * correct / total

    def save_checkpoint(self, path, epoch, val_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc
        }, path)
        logger.info(f"Saved model checkpoint to {path}")

    def save_class_names(self, path='car_model_classes.json'):
        with open(path, "w") as f:
            json.dump(self.class_names, f)
        logger.info(f"Saved class names to {path}")

    def train(self):
        best_acc = 0.0
        patience = 15
        no_improve = 0
        self.output_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            logger.info(f"[Epoch {epoch+1}/{self.num_epochs}] "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            self.scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(os.path.join(self.output_dir, "best_model.pth"), epoch, val_acc)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

        self.save_class_names()
        logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

    def test(self):
        """
        Evaluates the best model on the test set.
        """
        test_dir = os.path.join(self.data_dir, "test")
        if not os.path.exists(test_dir):
            logger.warning("Test directory not found. Skipping test evaluation.")
            return

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=os.cpu_count() if os.name != 'nt' else 0, pin_memory=True)

        checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning("Best model checkpoint not found. Cannot run test.")
            return

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        correct = 0
        total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = 100 * correct / total
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.class_names))))
        logger.info(f"Test Accuracy: {acc:.2f}%")
        logger.info(f"Test Confusion Matrix:\n{cm}")


def main():
    config = {
        'data_dir': "/mnt/d/Samart Eye/processed_dataset/finalize_data",
        'model_name': 'resnet50',
        'batch_size': 8,
        'num_epochs': 200,
        'lr': 3e-5
    }

    classifier = CarModelClassifier(**config)
    classifier.setup_data_loaders()
    classifier.setup_model()
    classifier.train()
    classifier.test() 


if __name__ == "__main__":
    main()
