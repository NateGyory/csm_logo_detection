import cv2
import numpy as np
import os
import sys
import torch
import torchvision
import csv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def load_images_and_labels(image_folder):
    label_dict = {}
    images_dict = {}

    # Load images
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder,filename))
        if img is not None:
            images_dict[filename] = img

    # Load labels
    with open(os.path.join(image_folder,'_annotations.csv')) as f:
        reader = csv.reader(f)
        flag = 1
        for row in reader:
            if flag:
                flag = 0
                continue
            x_min = int(row[4])
            y_min = int(row[5])
            x_max = int(row[6])
            y_max = int(row[7])
            box = [x_min, y_min, x_max, y_max]
            label_dict[row[0]] = box

    return images_dict, label_dict

def process_images(image_folder):
    images_dict, label_dict = load_images_and_labels(image_folder)
    images = []
    annotations = []

    for file_name, img in images_dict.items():
        # Conver cv2 to PIL image
        #box = label_dict[file_name]
        #start_point = (box[0], box[1])
        #end_point = (box[2], box[3])
        #image = cv2.rectangle(img, start_point, end_point, color = (255,0,0), thickness = 2)
        #cv2.imshow('TEST', image)
        #cv2.waitKey(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        images.append(img_pil)
        annotations.append(label_dict[file_name])

    return images, annotations

class LogoDataset(Dataset):
    def __init__(self, image_folder):
        self.imgs, self.annotations = process_images(image_folder)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        annotation = self.annotations[idx]

        boxes = torch.tensor([annotation], dtype=torch.float32)

        # Two labels: CSM Logo and background
        num_objs = 2
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # No crowds
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = torchvision.transforms.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model():
    if len(sys.argv) != 2:
        print('Please pass 1 image folder argument to the script')
        exit(0)

    model_exists = os.path.exists('./model.pth')
    if model_exists:
        print('Model is already trained, please run find_phone.py file to evaluate the model')
        exit(0)

    image_folder = sys.argv[1]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load custom Logo dataset
    dataset = LogoDataset(image_folder)

    data_loader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle = True,
        num_workers = 2,
        collate_fn = collate_fn
    )

    # Fine tune a pretrained faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 5

    itr = 1

    for epoch in range(num_epochs):
        for images, targets in data_loader:

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            print(f"Iterations #{itr} loss: {loss_value}")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iterations #{itr} loss: {loss_value}")

            itr += 1
            lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {loss_value}")

    torch.save(model.state_dict(), '../model/model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    },  '../model/ckpt.pth')

##########
#  Main  #
##########
train_model()
