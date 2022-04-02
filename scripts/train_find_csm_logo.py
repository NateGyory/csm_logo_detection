import cv2
import numpy as np
import os
import sys
import torch
import torchvision
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
    f = open(os.path.join(image_folder,'labels.txt'), 'r')
    lines = f.readlines()
    for line in lines:
        line_arr = line.split()
        label_dict[line_arr[0]] = line_arr[1:]

    return images_dict, label_dict

def process_images(image_folder):
    label_dict = {}
    images_dict, label_dict = load_images_and_labels(image_folder)
    images = []
    annotations = []

    for file_name, img in images_dict.items():
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img,(5,5),0)
        thr = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
        binary_img = cv2.bitwise_not(thr)

        # Opening and Closing morphology to remove noise
        ksize = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            x_phone = float(label_dict[file_name][0]) * img.shape[1]
            y_phone = float(label_dict[file_name][1]) * img.shape[0]

            # Determine if x,y is close enough to label x,y
            thresh = 7
            if abs(x_phone - (x + w/2)) < thresh and abs(y_phone - (y + h/2)) < thresh:
                dim_scalar = 14
                coord_scalar = 7
                w = dim_scalar + w
                h = dim_scalar + h
                x = x - coord_scalar
                y = y - coord_scalar

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_pil = Image.fromarray(img)
                images.append(img_pil)
                annotations.append([x, y, x+w, y+h])

    return images, annotations

class IphoneDataset(Dataset):
    def __init__(self, image_folder):
        self.imgs, self.annotations = process_images(image_folder)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        annotation = self.annotations[idx]

        boxes = torch.tensor([annotation], dtype=torch.float32)

        # Two labels: Iphone and background
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

    # Load custom Iphone dataset
    dataset = IphoneDataset(image_folder)

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

    num_epochs = 40

    itr = 1

    for epoch in range(num_epochs):
        for images, targets in data_loader:

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iterations #{itr} loss: {loss_value}")

            itr += 1
            lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {loss_value}")

    torch.save(model.state_dict(), './model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    },  './ckpt.pth')

##########
#  Main  #
##########
train_model()
