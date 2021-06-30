import pandas as pd
from scoring import Classes
from scoring import convert
import numpy as np
import cv2
import argparse
import glob
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scoring import get_arrow_coords
import torch
import copy
label_encoder = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10, "11":11, "12":12, "contour":13, "hand":14, "top_left_arrow":15, "top_right_arrow": 16, "right_arrow": 17, "left_arrow": 18, "up_arrow": 19, "down_arrow": 20, "bottom_right_arrow":21}
reflect_dct = {Classes.TOP_LEFT_ARROW: Classes.TOP_RIGHT_ARROW, Classes.TOP_RIGHT_ARROW: Classes.TOP_LEFT_ARROW, Classes.RIGHT_ARROW: Classes.LEFT_ARROW,
    Classes.LEFT_ARROW: Classes.RIGHT_ARROW, Classes.UP_ARROW: Classes.UP_ARROW, Classes.DOWN_ARROW: Classes.DOWN_ARROW, Classes.BOTTOM_LEFT_ARROW: Classes.BOTTOM_RIGHT_ARROW,
        Classes.RIGHT_ARROW: Classes.LEFT_ARROW}

class ImageAugmentGenerator(object):
    def __init__(self, transformation_prob = 0.5, rotation_angle = 30, rotation_prob = 0.5):
        self.transformation_prob = transformation_prob
        self.rotation_angle = rotation_angle
        self.rotation_prob = rotation_prob
        self.random_generator = np.random.default_rng(0)

    def transform_images(self, images, targets):
        # rotate digits and flip hands
        images_ = []
        targets_ = []
        rotation_dir = -1 # rotate clockwise
        if self.random_generator.random() > self.transformation_prob:
            if self.random_generator.random() > self.rotation_prob:
                # rotate clockwise
                rotation_angle = rotation_dir * self.rotation_angle
            else:
                rotation_angle = - rotation_dir * self.rotation_angle

            for image, target in zip(images, targets):
                image_ = image.copy()
                target_ = copy.deepcopy(target)
                self.transform_image_and_target(image_, target_, rotation_angle)
                images_.append(image_)
                targets_.append(target_)
        else:
            return images, targets

        return images_, targets_

    # transforms the image together with the bboxes. update bboxes in place
    def transform_image_and_target(self, image, target, rotation_angle):
        # not complete set of hand and sub hands.
        if len([i for i in target['labels'] if i > 13]) != 3:
            return

        clock_hand_at = (target['labels'] == 14).nonzero(as_tuple=True)[0].item()
        clock_hand_bbox = target['boxes'][clock_hand_at]
        clock_hand_bbox = clock_hand_bbox.int()
        clock_hand_x1, clock_hand_y1, clock_hand_x2, clock_hand_y2 = clock_hand_bbox[0], clock_hand_bbox[1], clock_hand_bbox[2], clock_hand_bbox[3]

        for index, (label, bbox) in enumerate(zip(target['labels'], target['boxes'])):
            if label == 14: # clock hand
                self.update_clock_hand_roi(image, clock_hand_bbox)
            elif label > 14:    # sub hand
                label_, bbox_ = self.update_clock_sub_hand_bbox_label(image, label, bbox, clock_hand_bbox)
                target['boxes'][index] = bbox_
                target['labels'][index] = label_
            elif label != 13:   # digit
                self.update_digit_roi(image, bbox, rotation_angle) # update roi for each digit. No change to bbox

    def update_clock_hand_roi(self, image, clock_hand_bbox):
        x1, y1, x2, y2 = clock_hand_bbox[0], clock_hand_bbox[1], clock_hand_bbox[2], clock_hand_bbox[3]
        hand_roi = image[y1:y2, x1:x2]
        M_roi = np.array([[-1, 0, hand_roi.shape[1]], [0, 1, 0]]).astype(np.float32)   # transformation to retrieve reflected hand image
        new_hand_im = cv2.warpAffine(hand_roi, M_roi, (hand_roi.shape[1], hand_roi.shape[0]), borderValue=[255, 255, 255])
        image[y1:y2, x1:x2] = new_hand_im

    def update_clock_sub_hand_bbox_label(self, image, label, sub_hand_bbox, clock_hand_bbox):
        x1, y1, x2, y2 = clock_hand_bbox[0], clock_hand_bbox[1], clock_hand_bbox[2], clock_hand_bbox[3]
        sub_hand_x1, sub_hand_y1, sub_hand_x2, sub_hand_y2 = sub_hand_bbox[0], sub_hand_bbox[1], sub_hand_bbox[2], sub_hand_bbox[3]

        sub_hand_x1_from_origin, sub_hand_x2_from_origin = sub_hand_x1 - x1, sub_hand_x2 - x1
        sub_hand_x1_from_origin_after_reflect, sub_hand_x2_from_origin_after_reflect = - sub_hand_x1_from_origin + (x2 - x1), - sub_hand_x2_from_origin + (x2 - x1)

        sub_hand_x1_, sub_hand_x2_ = sub_hand_x1_from_origin_after_reflect + x1, sub_hand_x2_from_origin_after_reflect + x1

        # reflected label
        label = label.item()
        label_ = reflect_dct[Classes(label)]
        label_ = label_.value

        return label_, torch.as_tensor([sub_hand_x2_, sub_hand_y1, sub_hand_x1_, sub_hand_y2], dtype=torch.float32)

    def update_digit_roi(self, image, digit_bbox, rotation_angle):
        x1,y1,x2,y2 = int(digit_bbox[0]), int(digit_bbox[1]), int(digit_bbox[2]), int(digit_bbox[3])
        digit_center_x, digit_center_y = (x1+x2) // 2, (y1 + y2) // 2
        digit_roi = image[y1:y2, x1:x2, :]
        H, W, _ = digit_roi.shape

        digit_roi_center_x = W // 2
        digit_roi_center_y = H // 2

        M = cv2.getRotationMatrix2D((digit_roi_center_x, digit_roi_center_y), rotation_angle, scale=1)

        new_roi = cv2.warpAffine(digit_roi, M, (W, H), borderValue = [255, 255, 255])

        image[y1:y2, x1: x2, :] = new_roi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./data_v1.1/train/train_images")
    parser.add_argument("--rotation_angle", type=str, default=15)
    parser.add_argument("--csv_path", type=str, default="./data_v1.1/train_labels.csv")
    parser.add_argument("--resize_height", type=int, default=800)
    parser.add_argument("--resize_width", type=int, default=800)
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    image_augmenter = ImageAugmentGenerator(transformation_prob = 0.5, rotation_angle = args.rotation_angle)

    images_batch = []
    targets_batch = []

    for im_path in glob.glob(args.image_dir + "/*.png"):
        im = cv2.imread(im_path)
        im_name = im_path.split("/")[-1]
        im_df = df[df['filename'] == im_name]
        im = cv2.resize(im, dsize=(args.resize_height, args.resize_width), interpolation=cv2.INTER_NEAREST)
        labels = []
        bboxes = []
        for _, row in im_df.iterrows():
            im_width = row['width']
            im_height = row['height']
            width_scale = args.resize_width / im_width
            height_scale = args.resize_height / im_height
            xmin = int(row['xmin'] * width_scale)
            xmax = int(row['xmax'] * width_scale)
            ymin = int(row['ymin'] * height_scale)
            ymax = int(row['ymax'] * height_scale)
            label = label_encoder[row['class']]
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        images_batch.append(im)
        targets_batch.append(target)

    i = 0
    while i < len(images_batch):
        images_mini_batch = images_batch[i:i+4]
        targets_mini_batch = targets_batch[i:i+4]
        transformed_images, transformed_targets = image_augmenter.transform_images(images_mini_batch, targets_mini_batch)
        i += 4
        fig, ax = plt.subplots(2, len(images_mini_batch))
        for j, (original_im, original_target) in enumerate(zip(images_mini_batch, targets_mini_batch)):
            bboxes = original_target['boxes']
            labels = original_target['labels']
            ax[0, j].imshow(original_im)
            for (label,bbox) in zip(labels, bboxes):
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
                ax[0, j].add_patch(rect)
                ax[0, j].text(bbox[0], bbox[1], label.item(), bbox=dict(facecolor='red', alpha=0.5))

        for j, (new_im, new_target) in enumerate(zip(transformed_images, transformed_targets)):
            bboxes = new_target['boxes']
            labels = new_target['labels']
            ax[1, j].imshow(new_im)
            for (label,bbox) in zip(labels, bboxes):
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
                ax[1, j].add_patch(rect)
                ax[1, j].text(bbox[0], bbox[1], label.item(), bbox=dict(facecolor='red', alpha=0.5))
        plt.show()



