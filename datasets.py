import torch
import cv2
import numpy as np
import os
import glob as glob

import json
from config import (
    CLASSES, RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform
import torchvision.transforms as transforms


# the dataset class
class EndometriosisDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format

        image = image.astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.json'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        with open(annot_file_path, 'r') as f:
            annot = json.load(f)
        allLabels = annot['label']
        allbb = annot['bb']

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        for indi in range(len(allLabels)):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(allLabels[indi]))

        # box coordinates for xml files are extracted and corrected for image size given
        for bb in allbb:
            # xmin = left corner x-coordinates
            xmin = bb[0][0]
            # xmax = right corner x-coordinates
            xmax = bb[1][0]
            # ymin = left corner y-coordinates
            ymin = bb[0][1]
            # ymax = right corner y-coordinates
            ymax = bb[1][1]

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            yamx_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def class_stats(ds):
    instances = [0] * len(CLASSES)
    frame_num = [0] * len(CLASSES)
    for ind in range(len(ds)):
        img, target = ds[ind]
        img_label = target['labels'].tolist()
        for cl in (img_label):
            instances[cl] += 1
        if instances[cl] > 0:
            frame_num[cl] = frame_num[cl] + 1
        print('image ' + str(ind))

    return instances, frame_num


# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = EndometriosisDataset(TRAIN_DIR, RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT, CLASSES, get_train_transform())
    return train_dataset


def create_valid_dataset():
    valid_dataset = EndometriosisDataset(VALID_DIR, RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT, CLASSES, get_train_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = EndometriosisDataset(
        TRAIN_DIR, RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    # instances, frame_num = class_stats(dataset)
    # print(instances)
    # print(frame_num)

    # print('Test Dataset')
    # DIR_TEST= 'data/Endo/test'
    # test_ds = EndometriosisDataset(
    #     DIR_TEST, RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT, CLASSES
    # )
    # instances, frame_num = class_stats(test_ds)
    # print(instances)
    # print(frame_num)
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 3
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
