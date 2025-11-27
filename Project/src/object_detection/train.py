import os
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
import sys

from dataset import KITTIDetectionDataset
from model import FasterRCNNLightning, YoloLightning

MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MODEL_NAME = "FasterRCNN"  # Options: "FasterRCNN", "YOLO"

WANDB_PROJECT_NAME = "object-detection-perception"


def collate_fn(batch):
    """
    Collate function to handle variable number of bounding boxes.
    :param batch: List of tuples containing an image and a target.
    :return: Tuple containing a tensor of images and a list of targets.
    """
    images, targets = [], []
    
    # Extract images and targets from batch
    for image, target in batch: 
        images.append(image)
        targets.append(target)
    
    # Don't stack the targets, just return them as a list
    return images, targets


def train(model, dataloaders):
    """
    Train the given model using the provided dataloaders.
    :param model: The model to train
    :param dataloaders: List of dataloaders [train, val, (test)]
    :return: Trained model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model.to(device)

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT_NAME, 
        name=f"{MODEL_NAME}_{wandb.util.generate_id()}", 
        log_model=False)
    
    # Log the model hyperparameters
    wandb_logger.log_hyperparams({
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "model_name": MODEL_NAME
    })
    # Set up early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_total_loss", patience=PATIENCE,
        verbose=True, mode="min"
    )
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator=device,
        callbacks=[early_stopping],
    )

    print("--- Starting Training ---")
    trainer.fit(model, dataloaders[0], dataloaders[1])
    print("--- Training Finished ---")
    
    if len(dataloaders) == 3:
        print("--- Starting Testing ---")
        trainer.test(model, dataloaders[2])
    
    return model


def save_model(model, save_model_dir):
    """
    Save the trained model to the specified directory with a timestamped filename.
    :param model: Trained model to save
    :param save_model_dir: Directory to save the model
    """

    # Create the save directory if it doesn't exist
    os.makedirs(save_model_dir, exist_ok=True) # Make sure the directory exists
    
    # Create a timestamped filename
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_filename = f"{model.name}-{date}.pth"
    save_path = os.path.join(save_model_dir, save_filename)

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


def create_datasets(data_path, train_split=0.8, test_split=0):
    """
    Create datasets for training, validation, and optionally testing.
    :param data_path: Path to the dataset
    :param train_split: Proportion of data to use for training
    :param test_split: Proportion of data to use for testing
    :return: Tuple of datasets (train, val, (test)) and number of classes
    """
    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    # Create the full dataset
    dataset = KITTIDetectionDataset(
        data_path=data_path,
        transforms=image_transforms
    )
    num_classes = dataset.get_num_classes()
    print(f"Number of classes in dataset: {num_classes}")
    print(f"Class names: {dataset.get_class_names()}")

    # Compute sizes of the trains, val, (test) splits
    total_size = len(dataset)
    train_size = int(train_split * total_size)

    # If test_split > 0, create train, val, test splits
    if test_split > 0:
        test_size = int(test_split * total_size)
        val_size = total_size - train_size - test_size
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

        # Split the dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        return (train_dataset, val_dataset, test_dataset), num_classes
    # Otherwise, create only train and val splits
    else:
        val_size = total_size - train_size
        print(f"Train size: {train_size}, Val size: {val_size}")

        # Split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        return (train_dataset, val_dataset), num_classes
    

def create_dataloaders(datasets, num_workers=4):
    """ 
    Create dataloaders for the given datasets.
    :param datasets: Tuple of datasets (train, val, (test))
    :param batch_size: Batch size for the dataloaders
    :param num_workers: Number of workers for data loading
    :return: List of dataloaders corresponding to the datasets
    """
    dataloaders = []
    # Create a dataloader for each dataset
    for dataset in datasets:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        dataloaders.append(dataloader)
    return dataloaders


def create_model(model_name, num_classes, learning_rate):
    """
    Create a model based on the specified name.
    :param model_name: Name of the model to create
    :param num_classes: Number of classes for the model
    :param learning_rate: Learning rate for the model
    :return: Initialized model
    """
    if model_name.lower() == "fasterrcnn":
        return FasterRCNNLightning(
            num_classes=num_classes,
            learning_rate=learning_rate,
            pretrained=True
        )
    elif model_name.lower() == "yolo":
        return YoloLightning(
            num_classes=num_classes,
            learning_rate=learning_rate,
            pretrained=True
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    data_path = "C:\\Users\\mjgoj\\Desktop\\Perception-AutonomousSystems\\Project\\data"
    
    # Create the datasets and dataloaders
    datasets, num_classes = create_datasets(data_path=data_path, train_split=0.8, test_split=0.1)
    dataloaders = create_dataloaders(datasets, num_workers=4)

    # Initialize the model
    model = create_model(MODEL_NAME, num_classes, LEARNING_RATE)

    trained_model = train(model, dataloaders)
    save_model(trained_model, 'trained_models/')

    