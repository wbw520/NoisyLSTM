import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from math import ceil
from parameter import args
from data_gen import MakeListSequence, MakeList, DataSet, DataSetSequence


def load_weight(model_pre, model_new):
    """
    load weight from pre-train parameter
    """
    model_pre_dict = model_pre.state_dict()
    model_new_dict = model_new.state_dict()
    model_pre_dict = {k: v for k, v in model_pre_dict.items() if k in model_new_dict}
    model_new_dict.update(model_pre_dict)
    model_new.load_state_dict(model_new_dict)


def fix_parameter(model, name_fix, mode="open"):
    """
    fix parameter for model training
    """
    for name, param in model.named_parameters():
        for i in range(len(name_fix)):
            if mode != "fix":
                if name_fix[i] not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    break
            else:
                if name_fix[i] in name:
                    param.requires_grad = False


def print_param(model):
    # show name of parameter could be trained in model
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def show_single(image):
    # show single image
    image = cv2.resize(image, (args.original_size[1], args.original_size[0]), interpolation=cv2.INTER_NEAREST)
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    plt.imshow(image)
    plt.axis('on')
    plt.show()


def load_data(lstm, random=True, batch_seq=args.sequence_len, batch_train=args.batch_size, use_noise=False):
    if lstm:
        L = MakeListSequence(args.data_dir, batch_seq, random=random).make_list()
        sequence_dataset = {"train": DataSetSequence(L["train"], use_noise=use_noise),
                       "val": DataSetSequence(L["val"], train=False, use_aug=False, use_noise=False)}
        dataloaders_lstm = {x: DataLoader(sequence_dataset[x], batch_size=batch_train//batch_seq, shuffle=False, num_workers=4)
                            for x in ["train", "val"]}
        print("load lstm data over")
        return dataloaders_lstm
    else:
        L = MakeList(args.data_dir).make_list()
        image_dataset = {"train": DataSet(L["train"]),
                         "val": DataSet(L["val"], train=False, use_aug=False)}
        dataloaders = {x: DataLoader(image_dataset[x], batch_size=batch_train, shuffle=True, num_workers=4)
                       for x in ["train", "val"]}
        print("load normal data over")
        return dataloaders


class IouCal(object):
    def __init__(self, num_class=args.num_classes):
        self.num_class = num_class
        self.hist = np.zeros((self.num_class, self.num_class))
        self.name = ["road:", "sidewalk:", "building:", "wall:", "fence:", "pole:", "traffic light:", "traffic sign:",
                     "vegetation:", "terrain:", "sky:", "person:", "rider:", "car:", "truck:", "bus:", "train:",
                     "motorcycle:", "bicycle:"]

    def fast_hist(self, label, pred, num_class):
        k = (label < self.num_class) & (pred < self.num_class)
        return np.bincount(num_class * label[k].astype(int) + pred[k], minlength=num_class ** 2).reshape(num_class, num_class)

    def per_class_iou(self, hist):
        return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))   # IOU = TP / (TP + FP + FN)

    def evaluate(self, labels, preds):
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())
        for label, pred in zip(labels, preds):
            self.hist += self.fast_hist(label.flatten(), pred.flatten(), self.num_class)

    def iou_demo(self):
        iou = self.per_class_iou(self.hist)
        STR = ""
        for i in range(len(self.name)):
            STR = STR + self.name[i] + str(round(iou[i], 3)) + " "
        print(STR)
        miou = np.nanmean(iou)
        return round(miou, 3)


class ColorTransition(object):
    def __init__(self, ignore_label=19):
        self.root = "/home/wbw/PycharmProjects/city/gtFine/train/"
        self.root_new = "/home/wbw/PycharmProjects/city_hehe/train/"
        self.color = [[128, 64, 128],   # class 0   road
                      [244, 35, 232],   # class 1   sidewalk
                      [70, 70, 70],     # class 2   building
                      [102, 102, 156],  # class 3   wall
                      [190, 153, 153],  # class 4   fence
                      [153, 153, 153],  # class 5   pole
                      [250, 170, 30],   # class 6   traffic light
                      [220, 220, 0],    # class 7   traffic sign
                      [107, 142, 35],   # class 8   vegetation
                      [152, 251, 152],  # class 9   terrain
                      [70, 130, 180],   # class 10  sky
                      [220, 20, 60],    # class 11  person
                      [255, 0, 0],      # class 12  rider
                      [0, 0, 142],      # class 13  car
                      [0, 0, 70],       # class 14  truck
                      [0, 60, 100],     # class 15  bus
                      [0, 80, 100],     # class 16  train
                      [0, 0, 230],      # class 17  motorcycle
                      [119, 11, 32],    # class 18  bicycle
                      [0, 0, 0]]        # class 19  background
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                          3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                          7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                          14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                          18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                          28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def recover(self, image):  # convert predict of binary to color
        h, w = image.shape
        color_image = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                color_image[i][j] = self.color[int(image[i][j])]
        return color_image.astype(np.uint8)

    # trainslate label_id to train_id
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, crop_size, classes, lstm):
    image_size = image.size()
    tile_rows = ceil(image_size[2]/crop_size[0])
    tile_cols = ceil(image_size[3]/crop_size[1])
    stride_rows = crop_size[0] - (crop_size[0]*tile_rows - image_size[2])//(tile_rows-1)
    stride_cols = crop_size[1] - (crop_size[1]*tile_cols - image_size[3])//(tile_cols-1)
    if not lstm:
        batch = image_size[0]
    else:
        batch = image_size[0]//args.sequence_len
    full_probs = torch.from_numpy(np.zeros((batch, classes, image_size[2], image_size[3]))).to(args.gpu)
    count_predictions = torch.from_numpy(np.zeros((batch, classes, image_size[2], image_size[3]))).to(args.gpu)

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride_cols)
            y1 = int(row * stride_rows)
            x2 = x1 + crop_size[1]
            y2 = y1 + crop_size[0]
            if row == tile_rows - 1:
                y2 = image_size[2]
                y1 = image_size[2] - crop_size[0]
            if col == tile_cols - 1:
                x2 = image_size[3]
                x1 = image_size[3] - crop_size[1]

            # print(x1, x2, y1, y2)
            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                padded_prediction = net(img)
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += padded_prediction  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    _, preds = torch.max(full_probs, 1)
    return preds