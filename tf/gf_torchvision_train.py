# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import xml.etree.ElementTree as ET

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import cv2
from pycocotools import mask as maskUtils

import wwtool

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labelxmls"))))


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labelxmls", self.labels[idx])
        #img=cv2.imread(img_path)
        #img=img[...,::-1]
        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(label_path)
        root = tree.getroot()
        objects = []

        imgsize=root.find('size')
        width = int(imgsize.find('width').text)
        height = int(imgsize.find('height').text)

        boxes = []
        thetaobbes = []
        pointobbes = []
        masks = []
        for single_object in root.findall('object'):
            robndbox = single_object.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w  = float(robndbox.find('w').text)
            h  = float(robndbox.find('h').text)
            theta = float(robndbox.find('angle').text)
            thetaobb = [cx,cy,w,h,theta]


            box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
            box = np.reshape(box, [-1, ]).tolist()
            pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

            xmin = min(pointobb[0::2])
            ymin = min(pointobb[1::2])
            xmax = max(pointobb[0::2])
            ymax = max(pointobb[1::2])
            bbox = [xmin, ymin, xmax, ymax]
            
            reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            reference_bbox = np.array(reference_bbox)
            normalize = np.array([1.0, 1.0] * 4)
            combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
            distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
            sorted = distances.argsort()
            pointobb = combinate[sorted[0]].tolist()

            thetaobbes.append(thetaobb)
            pointobbes.append(pointobb)
            boxes.append(bbox)

            segm = [pointobb]            
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)

            masks.append(mask)

        
        num_objs = len(masks)
        thetaobbes = torch.as_tensor(thetaobbes, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        #wwtool.imshow_bboxes(img_path,boxes.detach().numpy(),labels.detach().numpy())

        image_id = torch.tensor([idx])
        area = thetaobbes[:, 2]*thetaobbes[:,3]
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        #print(img[0, 0, 0:10])
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_object_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # now get the number of inputfeatures for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(root_dir, model_save_dir):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2


    # use our dataset and defined transformations
    dataset = PennFudanDataset(root_dir, get_transform(train=True))
    dataset_test = PennFudanDataset(root_dir, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_object_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0001)
    # and a learning rate scheduler
    '''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    '''
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[8,11],
                                                   gamma=0.1)    
    # let's train it for 10 epochs
    num_epochs = 12

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        torch.save(model.state_dict(), model_save_dir + '/{}.pth'.format(epoch))
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
