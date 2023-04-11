import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import json
from model import create_model
import numpy as np
import pygsheets
import pandas as pd
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)


def write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, data_df):
    """
    this function takes data_df and writes it under spreadsheet_id
    and sheet_name using your credentials under service_file_path
    """
    gc = pygsheets.authorize(service_file=service_file_path)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        sh.add_worksheet(sheet_name)
    except:
        pass
    wks_write = sh.worksheet_by_title(sheet_name)
    wks_write.clear('A1', None, '*')
    wks_write.set_dataframe(data_df, (1, 1), encoding='utf-8', fit=True)
    wks_write.frozen_rows = 1


def compute_dice_mask(predicted_labels_list, predicted_boxes_list, ground_truth_labels_list, ground_truth_boxes_list,
                      print_box):
    # Get the number of classes in the dataset
    num_classes = NUM_CLASSES
    dice_scores = [None] * num_classes
    predicted_boxes = []
    ground_truth_boxes = []
    # Iterate over each class
    for c in range(1, num_classes):
        # Get the predicted boxes and ground truth boxes for class c

        if len(predicted_labels_list) != 0:
            for indlabel in range(len(predicted_labels_list)):
                if predicted_labels_list[indlabel] == CLASSES[c]:
                    predicted_boxes.append(predicted_boxes_list[indlabel])

        if len(ground_truth_labels_list) != 0:
            for indlabel in range(len(ground_truth_labels_list)):
                if ground_truth_labels_list[indlabel] == CLASSES[c]:
                    ground_truth_boxes.append(ground_truth_boxes_list[indlabel])

        # if there are no class c in the ground truth , and no prediction : TN
        if len(ground_truth_boxes) == 0 and len(predicted_boxes) == 0:
            # print('True Negatif')
            dice_score = 'nan'
        # if there are no class c in the ground truth , or no prediction : it can be FN, or FP
        elif len(ground_truth_boxes) == 0 or len(predicted_boxes) == 0:
            dice_score = 0

        else:
            # Compute the dice score for class c for TP
            mask_pred = create_mask_bbox(predicted_boxes)
            mask_GT = create_mask_bbox(ground_truth_boxes)
            if print_box:
                indices = mask_GT.astype(np.uint8)  # convert to an unsigned byte
                indices *= 255
                cv2.imshow('Indices', indices)
                cv2.waitKey(0)
                indices = mask_pred.astype(np.uint8)  # convert to an unsigned byte
                indices *= 255
                cv2.imshow('Indices', indices)
                cv2.waitKey(0)
            dice_score = dice(mask_pred, mask_GT)
        predicted_boxes = []
        ground_truth_boxes = []
        # Store the dice score in the dictionary
        dice_scores[c] = dice_score

    return dice_scores


def create_mask_bbox(bboxes):
    mask = np.full((size_img[0], size_img[1]), False)  # initialize mask
    for box in bboxes:
        mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def dice(mask1, mask2):
    m1 = mask1.flatten()
    m2 = mask2.flatten()
    score = 2 * sum(np.logical_and(m1, m2)) / (sum(m1) + sum(m2))
    return score


files = glob.glob('"inference_outputs/images/*')
for f in files:
    os.remove(f)
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# directory where all the images are present
DIR_TEST = 'data/Endo/test'
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.6

# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0
all_dices = [[] * (NUM_CLASSES + 1) for _ in
             range(len(test_images))]  # This contqins the Dice for every class for every Image
for i in range(len(test_images)):  # range(10):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0] + '.' + \
                 test_images[i].split(os.path.sep)[-1].split('.')[1]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    with open(test_images[i].replace('jpg', 'json'), 'r') as f:
        annot = json.load(f)
    allLabels = annot['label']
    allbb = annot['bb']
    size_img = image.shape
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1

    # draw the original bounding boxes and write the class name on top of it
    for indi in range(len(allLabels)):
        bb = allbb[indi]

        cv2.rectangle(orig_image,
                      (bb[0][0], bb[0][1]),
                      (bb[1][0], bb[1][1]),
                      [255, 0, 0], 3)
        cv2.putText(orig_image, allLabels[indi],
                    (bb[0][0], bb[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255, 0, 0],
                    2, lineType=cv2.LINE_AA)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    classes_predicted = []
    boxes_predicted = []
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()

        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        # print(pred_classes)
        # print(scores)
        # draw the bounding boxes and write the class name on top of it

        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            classes_predicted.append(class_name)
            boxes_predicted.append(box)
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)
            cv2.putText(orig_image, class_name + ' ' + str(round(scores[j], 2)),
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                        2, lineType=cv2.LINE_AA)

    # cv2.imshow('Prediction', orig_image)
    # cv2.waitKey()

    GTboxes = [[box[0][0], box[0][1], box[1][0], box[1][1]] for box in allbb]
    all_dices[i].append(image_name)
    all_dices[i][1:] = compute_dice_mask(classes_predicted, boxes_predicted, allLabels, GTboxes, False)

    cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i + 1} done...")
    print('-' * 50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

sfpath = 'keycode/my-gpysheets-3d8d13442005.json'
sheetID = '13Dd30LxaBXVaBpeBXmVboZzxhysJd-ep5m6l3fIooMM'
sheetName = 'FASTER-RCNN ' + str(detection_threshold)+ ' with TN'

cols = ['Video Name']
cols += CLASSES
# write_to_gsheet(sfpath, sheetID, sheetName, pd.DataFrame(all_dices, columns=cols))
