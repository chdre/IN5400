import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Additional/supporting methods and code for the 2020 IN5400 lab on object classification and localization.
# Some of this code has been copied or heavily inspired by chunks found online.

def bb_intersection_over_union(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou  # return the intersection over union value


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_IoUs(preds, targets):
    ious = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        ious[i] = bb_intersection_over_union(preds[i, :], targets[i, :])
    return ious


def imshow(img, gt_box, pred_box=None):
    # Revert the color scaling done when loading the images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(std * img + mean, 0, 1)

    plt.imshow(img)

    def draw_box(box, color='green'):
        x = box[0] * img.shape[0]
        y = box[1] * img.shape[1]
        w = (box[2] - box[0]) * img.shape[0]
        h = (box[3] - box[1]) * img.shape[1]
        plt.gca().add_patch(plt.Rectangle((y, x), h, w, fill=False, edgecolor=color, linewidth=2, alpha=0.5))

    draw_box(gt_box)
    if pred_box is not None:
        draw_box(pred_box, 'red')


class CUBTypeDataset(Dataset):
    def __init__(self, im_ids, data_dir, transform=None):
        data_images_txt = os.path.join(data_dir, 'images.txt')
        data_bounding_boxes_txt = os.path.join(data_dir, 'bounding_boxes.txt')
        data_images = os.path.join(data_dir, 'images')
        with open(data_images_txt) as f:
            id_to_path = dict([l.split(' ', 1) for l in f.read().splitlines()])
        with open(data_bounding_boxes_txt) as f:
            id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                id_to_box[im_id] = list(map(float, box))
        self.imgs = [(os.path.join(data_images, id_to_path[i]), id_to_box[i])
                     for i in im_ids]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        path, box = self.imgs[index]
        im = Image.open(path).convert('RGB')
        im_size = np.array(np.flip(im.size), dtype='float32')
        box = np.array(box, dtype='float32')

        if not np.all(box == -1):
            box = np.concatenate([box[[1, 0]], box[[1, 0]] + box[[3, 2]]])  # [x0,y0,x1,y2]
            box = box / im_size[[0, 1, 0, 1]]  # Make coords of box relative to image size

        # Here, one might want to augment the data by randomly flipping the images, adjusting their colors/intensities,
        # cropping them, or the like.

        box = torch.Tensor(box)
        im = self.transform(im)
        return im, box, im_size

    def __len__(self):
        return len(self.imgs)


def split_cub_set(ratio, data_dir):
    # Split the ID's into train and val with a ratio of "ratio" in the validation set.  Each subclass is treated separately.
    data_images_txt = os.path.join(data_dir, 'images.txt')
    with open(data_images_txt) as f:
        lines = f.read().splitlines()
    class_groups = dict()
    for line in lines:
        value, line = line.split(' ', 1)
        key = line.split('.', 1)[0]
        value = value
        if key in class_groups:
            class_groups[key].append(value)
        else:
            class_groups[key] = [value]

    val_id = []
    for _, group in class_groups.items():
        val_id.extend(random.sample(group, int(math.ceil(len(group) * ratio))))
    train_id = [i for i in map(str, range(1, len(lines) + 1)) if i not in val_id]

    return train_id, val_id


