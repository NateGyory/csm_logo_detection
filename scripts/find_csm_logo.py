import cv2
import numpy as np
import os
import sys
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import torch
import torchvision
import warnings

# Need to ignore Pytorch deprication warning so as not to mess up the output
warnings.filterwarnings("ignore")

def load_image(image_path):
    images = []
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        images.append(img_pil)
    else:
        print('You did specify a valid image file')
        exit(0)

    return images

class IphoneDataset(Dataset):
    def __init__(self, image_path):
        self.imgs = load_image(image_path)

    def __getitem__(self, idx):
        img = self.imgs[idx]

        img = torchvision.transforms.ToTensor()(img)

        return img

    def __len__(self):
        return len(self.imgs)

def get_coordinates(box, img):
    h, w,_ = img.shape
    x_center = int((box[2] - box[0]) / 2 + box[0])
    y_center = int((box[3] - box[1]) / 2 + box[1])
    x_norm = x_center / w
    y_norm = y_center / h

    # Draw center of iphone on the image
    cv2.circle(img, (x_center, y_center), radius=0, color=(36,255,12), thickness=5)
    return [x_norm, y_norm]

def find_phone():
    if len(sys.argv) != 2:
        print('Please pass 1 image path argument to the script')
        exit(0)

    image_file = sys.argv[1]

    dataset = IphoneDataset(image_file)
    data_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 1
    )

    # Load the model and set it to eval mode
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    # Evaluate the image
    images_iter = iter(data_loader)
    for i in range(len(images_iter)):
        images = next(images_iter)
        images = list(image for image in images)
        outputs = model(images)

        transforms = torchvision.transforms.ToPILImage()
        im = transforms(images[0])
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        boxes = outputs[0]['boxes']
        score = outputs[0]['scores']

        box = boxes[0]
        normalized_coordinates = get_coordinates(box, img)
        norm_string = '({:.4f}, {:.4f})'.format(normalized_coordinates[0], normalized_coordinates[1])

        # Draw Iphone bounding box and place the Normalized Coordinates onto the images top left corner
        cv2.putText(img, norm_string, (1,29), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36,255,12), 2)

        print('{:.4f} {:.4f}'.format(normalized_coordinates[0], normalized_coordinates[1]))

        cv2.imshow("Find Phone", img)
        cv2.waitKey(0)

##########
#  Main  #
##########
find_phone()
