import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import torch
import torch.nn as nn
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import ConnectionPatch
import torch.nn.functional as F
from models.DeepLab.deeplab import DeepLab
from models.FCN import FCN8s, VGGNet
from models.DANET.danet import get_danet
from models.ICNet.icnet import ICNet
from models.PSPNet import PspNet


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


def show_single(image, location=None, save=False):
    # show single image
    image = np.array(image, dtype=np.uint8)
    # image = cv2.resize(image, (args.original_size[1], args.original_size[0]), interpolation=cv2.INTER_NEAREST)
    # fig, ax = plt.subplots(1, 1)
    # axins1 = ax.inset_axes((0.2, 0.05, 0.3, 0.3))
    # axins2 = ax.inset_axes((0.6, 0.2, 0.3, 0.3))
    # axins3 = ax.inset_axes((0.25, 0.7, 0.25, 0.25))
    # make_da(210, 440, 350, 600, image, axins1, ax)
    # make_da2(1220, 1380, 210, 320, image, axins2, ax)
    # make_da3(630, 770, 430, 570, image, axins3, ax)
    # plt.xticks([])
    # plt.yticks([])
    plt.imshow(image)
    # fig.set_size_inches(2048/100.0, 1024/100.0) #输出width*height像素
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if save:
        plt.savefig("imgs/"+location+".png", bbox_inches='tight', pad_inches=0)
    plt.show()


# #for drawing magnify images
# def make_da(xlim0, xlim1, ylim0, ylim1, image, axins, ax):
#     axins.imshow(image)
#     axins.set_xlim(xlim0, xlim1)
#     axins.set_ylim(ylim1, ylim0)
#     axins.axis("off")
#     tx0 = xlim0
#     tx1 = xlim1
#     ty0 = ylim0
#     ty1 = ylim1
#     sx = [tx0, tx1, tx1, tx0, tx0]
#     sy = [ty0, ty0, ty1, ty1, ty0]
#     ax.plot(sx, sy, "black", linewidth="4")
#
#     xy = (xlim1, ylim0)
#     xy2 = (xlim0, ylim0)
#     con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                           axesA=axins, axesB=ax, linewidth="4")
#     axins.add_artist(con)
#
#     xy = (xlim1, ylim1)
#     xy2 = (xlim0, ylim1)
#     con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                           axesA=axins, axesB=ax, linewidth="4")
#     axins.add_artist(con)


class IouCal(object):
    def __init__(self, args):
        self.num_class = args.num_classes
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
        self.color = {0: [128, 64, 128],   # class 0   road
                      1: [244, 35, 232],   # class 1   sidewalk
                      2: [70, 70, 70],     # class 2   building
                      3: [102, 102, 156],  # class 3   wall
                      4: [190, 153, 153],  # class 4   fence
                      5: [153, 153, 153],  # class 5   pole
                      6: [250, 170, 30],   # class 6   traffic light
                      7: [220, 220, 0],    # class 7   traffic sign
                      8: [107, 142, 35],   # class 8   vegetation
                      9: [152, 251, 152],  # class 9   terrain
                      10: [70, 130, 180],   # class 10  sky
                      11: [220, 20, 60],    # class 11  person
                      12: [255, 0, 0],      # class 12  rider
                      13: [0, 0, 142],      # class 13  car
                      14: [0, 0, 70],       # class 14  truck
                      15: [0, 60, 100],     # class 15  bus
                      16: [0, 80, 100],     # class 16  train
                      17: [0, 0, 230],      # class 17  motorcycle
                      18: [119, 11, 32],    # class 18  bicycle
                      19: [0, 0, 0]}        # class 19  background

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                          3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                          7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                          14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                          18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                          28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def recover(self, image):  # convert predict of binary to color
        image = image.cpu().detach().numpy()
        color_image = self.id2trainId(image, reverse=True)
        return color_image.astype(np.uint8)

    # trainslate label_id to train_id for color img
    def id2trainId(self, label, reverse=False):
        if reverse:
            w, h = label.shape
            label_copy = np.zeros((w, h, 3), dtype=np.uint8)
            for index, color in self.color.items():
                label_copy[label == index] = color
        else:
            w, h, c = label.shape
            label_copy = np.zeros((w, h), dtype=np.uint8)
            for index, color in self.color.items():
                label_copy[np.logical_and(*list([label[:, :, i] == color[i] for i in range(3)]))] = index
        return label_copy

    # trainslate label_id to train_id for binary img
    def id2trainIdbinary(self, label, reverse=False):
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


def predict_sliding(args, net, image, crop_size, classes, lstm):
    image_size = image.size()
    tile_rows = ceil(image_size[2]/crop_size[0])
    tile_cols = ceil(image_size[3]/crop_size[1])
    stride_rows = crop_size[0] - (crop_size[0]*tile_rows - image_size[2])//(tile_rows-1)
    stride_cols = crop_size[1] - (crop_size[1]*tile_cols - image_size[3])//(tile_cols-1)
    if not lstm or args.show_sequence:
        batch = image_size[0]
    else:
        batch = image_size[0]//args.sequence_len
    full_probs = torch.from_numpy(np.zeros((batch, classes, image_size[2], image_size[3]))).to(args.device)
    count_predictions = torch.from_numpy(np.zeros((batch, classes, image_size[2], image_size[3]))).to(args.device)

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

            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                padded_prediction = net(img)
                if isinstance(padded_prediction, tuple):
                    padded_prediction = padded_prediction[0]
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += padded_prediction  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    _, preds = torch.max(full_probs, 1)
    return preds


"""Custom losses."""
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, preds, target):
        pred, pred_sub4, pred_sub8, pred_sub16 = preds
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight


def load_model(args):
    if args.model_name == "ICNet":
        init_model = ICNet(args, use_lstm=args.lstm)
    else:
        init_model = PspNet(args, use_lstm=args.lstm)

    if args.use_pre:
        if args.model_name == "ICNet":
            pre_model = ICNet(args, use_lstm=False)
            pre_name = "ICNet"
        else:
            pre_model = PspNet(args, use_aux=False, use_lstm=False)
            pre_name = "PSPNet"
        pre_model.load_state_dict(torch.load("saved_model/" + pre_name + ".pt"), strict=True)
        print("load pre-train param over")
        load_weight(pre_model, init_model)
        # fix_parameter(init_model, ["final1", "final2", "CON_ls"])
        # print("param could be trained")
        # print_param(init_model)
    return init_model