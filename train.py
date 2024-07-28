import os
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex
from torchvision import transforms
from tqdm import tqdm

# Log into Wandb
wandb.login()

class AI4MARSDataset(Dataset):
    def __init__(self, images_path: str, masks_path: str, dataset_size: int = 500):
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_size = dataset_size
        
        images = set(os.listdir(images_path))
        self.masks = [mask for mask in os.listdir(masks_path) if mask[:-4] + ".JPG" in images][:dataset_size]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_name = self.masks[idx]

        image_path = os.path.join(self.images_path, mask_name[:-4] + ".JPG")
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))  # Changed to 256x256 for U-Net
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Change the order of dimensions to (C, H, W)
        image = torch.from_numpy(image)

        mask_path = os.path.join(self.masks_path, mask_name)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # Changed to 256x256 for U-Net
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 255] = 4
        mask = torch.from_numpy(mask)
        mask = mask.long()

        return image, mask

class AI4MARSDataModule(pl.LightningDataModule):
    def __init__(self, images_path: str, masks_path: str, dataset_size: int = 5000, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = AI4MARSDataset(self.images_path, self.masks_path, self.dataset_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.dc1 = double_conv(n_channels, 64)
        self.dc2 = double_conv(64, 128)
        self.dc3 = double_conv(128, 256)
        self.dc4 = double_conv(256, 512)
        self.dc5 = double_conv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dc6 = double_conv(1024, 512)
        self.dc7 = double_conv(512, 256)
        self.dc8 = double_conv(256, 128)
        self.dc9 = double_conv(128, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(nn.functional.max_pool2d(x1, 2))
        x3 = self.dc3(nn.functional.max_pool2d(x2, 2))
        x4 = self.dc4(nn.functional.max_pool2d(x3, 2))
        x5 = self.dc5(nn.functional.max_pool2d(x4, 2))

        x = self.up1(x5)
        x = self.dc6(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.dc7(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.dc8(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.dc9(torch.cat([x1, x], dim=1))
        
        return self.final(x)

class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model = UNet(n_channels=3, n_classes=num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, y), on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)

def train_single_run(data_module: AI4MARSDataModule, epochs: int = 10):
    # Initialize wandb
    wandb.init(project="AI4MARS_UNet", name="UNet_run", config={
        "learning_rate": 1e-4,
        "epochs": epochs,
        "batch_size": data_module.batch_size,
        "model": "U-Net",
        "dataset": "AI4MARS",
        "dataset_size": data_module.dataset_size
    })

    # Split the dataset into training and validation sets
    dataset_size = len(data_module.dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(data_module.dataset, [train_size, val_size])

    # Set up data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=data_module.batch_size, shuffle=True, num_workers=data_module.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=data_module.batch_size, num_workers=data_module.num_workers)

    # Initialize the Lightning model
    model = ImageSegmentationModel()

    # Set up Weights & Biases logger
    wandb_logger = WandbLogger(project="AI4MARS_UNet", name="UNet_run")

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # Set up the Lightning trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50
    )

    # Train and validate the model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save the model checkpoint after training
    torch.save(model.state_dict(), "trained_unet_model_checkpoint.pth")

    # End current run
    wandb.finish()
    
    # Return summary
    return wandb_logger.experiment.summary


# Execution
# Paths (adjust these to match your local file structure)
IMAGES_PATH = "/data/ai4mars-dataset-merged-0.1/msl/images/edr"
MASK_PATH_TRAIN = "/data/ai4mars-dataset-merged-0.1/msl/labels/train"
MASK_PATH_TEST = "/data/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree"

# Hyperparameters
DATASET_SIZE = 15000  # Adjust this based on your available memory and compute power
BATCH_SIZE = 8  # Adjust based on your GPU memory
EPOCHS = 15  # Adjust based on your needs and available time

# Load dataset
rock_data = AI4MARSDataModule(IMAGES_PATH, MASK_PATH_TRAIN, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, num_workers=2)
rock_data.setup()

# Train model
summary = train_single_run(rock_data, epochs=EPOCHS)

# Choose device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the saved model
model = ImageSegmentationModel()
model.load_state_dict(torch.load("trained_unet_model_checkpoint.pth"))
model.eval()
model.to(device)

print("Training completed and model loaded successfully.")