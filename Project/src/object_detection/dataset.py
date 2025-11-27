import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys


class KITTIDetectionDataset(Dataset):

    def __init__(self, data_path, transforms=None):
        super().__init__()
        
        self.transforms = transforms
        self.data = self.load_data(data_path)
        # mapping KITTI class names → integer labels
        self.class_to_idx = {cls: i+1 for i, cls in enumerate(self.get_class_names())}
        
    def load_data(self, data_path):
        """
        Load KITTI dataset from the specified path.
        :param data_path: Path to the KITTI label files
        :return: List of dictionaries with image names, classes, and bounding boxes
        """

        images_path = Path(data_path) / "images" / "training"
        labels_path = Path(data_path) / "labels"

        # Collect data from all KITTI labels
        label_files = sorted(labels_path.glob("*.txt"))
        print(f"Found {len(label_files)} label files.")
        data = []

        # Parse each label file
        for label_path in tqdm(label_files, desc="Parsing KITTI labels", unit="file"):
            image_name = images_path / (label_path.stem + ".png")
            if not image_name.exists():
                continue
            objects = self._parse_label(label_path)

            # Extract boxes and labels for this image
            boxes, labels = [], []
            for cls, x1, y1, x2, y2 in objects:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

            data.append({
                "image": images_path / image_name,
                "boxes": boxes,
                "labels": labels
            })
        return data

    def _parse_label(self, label_path):
        """ 
        Parse a KITTI label file to extract object classes and bounding boxes.
        :param label_path: Path to the KITTI label file
        :return: List of tuples (class, x1, y1, x2, y2)
        """
        objects = []
        with open(label_path, "r") as f:
            # Each line corresponds to one object
            for line in f:
                object_info = line.strip().split()
                if len(object_info) < 8:
                    continue
                cls = object_info[0]
                x1, y1, x2, y2 = map(float, object_info[4:8])
                objects.append((cls, x1, y1, x2, y2))
        return objects
    
    def get_class_names(self):
        """
        Get the list of unique class names in the dataset.
        :return: List of class names
        """
        class_names = set()
        for item in self.data:
            for cls in item["labels"]:
                class_names.add(cls)
        return sorted(list(class_names))

    def get_num_classes(self):
        """
        Get the number of unique classes in the dataset.
        :return: Number of classes
        """
        return len(self.get_class_names()) + 1  # +1 for background class

    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        :return: Dictionary with class names as keys and their counts as values
        """
        class_names = self.get_class_names()
        distribution = {cls: 0 for cls in class_names}
        for item in self.data:
            for cls in item["labels"]:
                distribution[cls] += 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        boxes = torch.tensor(item["boxes"], dtype=torch.float32)
        labels = torch.tensor(
            [self.class_to_idx[c] for c in item["labels"]],
            dtype=torch.int64
        )

        if self.transforms:
            image = self.transforms(image)

        # torchvision / YOLO compatible target format
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target
    
    def plot_sample(self, idx, ax=None):
        """
        Plot a sample image with bounding boxes.
        :param idx: Index of the sample to plot
        """

        image, target = self[idx]
        boxes = target["boxes"]
        labels = target["labels"]
        class_names = self.get_class_names()
        if ax is None:
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            ax = plt.gca()
        else:
            ax.imshow(image)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 10,
                class_names[label - 1],
                color='yellow', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5)
            )   
        ax.set_axis_off()


if __name__ == "__main__":
    data_path = "C:\\Users\\mjgoj\\Desktop\\Perception-AutonomousSystems\\Project\\data"
    dataset = KITTIDetectionDataset(data_path=data_path)

    print("Number of classes:", dataset.get_num_classes())
    print("Class distribution:", dataset.get_class_distribution())

    # Plot a sample
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax = ax.flatten()
    for i in range(6):
        dataset.plot_sample(i, ax=ax[i])
    plt.tight_layout()
    plt.show()