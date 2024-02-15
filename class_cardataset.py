import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os 
import torch

class CarDataset(Dataset):
    def __init__(self, dataset_root_dir, transform=None, mode="train"):
        self.transform = transform
        if mode == "train":
            self.dataset_name = "train"
        elif mode == "val":
            self.dataset_name = "val"
        elif mode == "test":
            self.dataset_name = "test"
        else:
            print("wrong dataset mode, please check")
        self.image_files_list = list(sorted(os.listdir(os.path.join(dataset_root_dir, "images", self.dataset_name))))
        self.label_files_list = list(sorted(os.listdir(os.path.join(dataset_root_dir, "labels", self.dataset_name))))
        self.dataset_root_dir = dataset_root_dir

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        # load image labels
        image_path = os.path.join(self.dataset_root_dir, "images", self.dataset_name, self.image_files_list[idx])
        annotation_path = os.path.join(self.dataset_root_dir, "labels", self.dataset_name, self.image_files_list[idx].replace('.png', '.txt'))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # sparse label files
        boxes = []
        with open(annotation_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts[0] == 'Car':
                    x_min, y_min, x_max, y_max = map(int, parts[1:])
                    boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target