from tools.tool import ColorTransition, IouCal, show_single, load_model
from train import get_args_parser
import argparse
from engine import for_val, for_test
import cv2
from tools.data_gen import niuqu, gauss
import torch.nn as nn
from tools.sync_batchnorm import convert_model
import numpy as np
import torch


mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
root_image = "/home/wbw/val/munster/"
root_label = "/home/wbw/gtFine/val/munster/"
image_name = "munster_000055_"
nm = 16  # start frame of the video


def make_image(model, img_name, device):
    print(image_name)
    image = cv2.imread(img_name, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (2048, 1024), interpolation=cv2.INTER_LINEAR)
    image = np.array(image, dtype=np.float32)
    image -= mean
    images = torch.from_numpy(np.array(np.transpose([image], (0, 3, 1, 2)), dtype="float32")/255)
    inputs = images.to(device, dtype=torch.float32)

    predict = for_test(args, model, inputs, None, None, lstm=False, need=True)
    color_img = ColorTransition().recover(torch.squeeze(predict, dim=0))
    # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    show_single(args, color_img, "base")


def make_image_lstm(model, device):
    iou = IouCal(args)
    images = []
    for i in range(4):  # extract continues four frames with skip 1
        image = cv2.imread(root_image + image_name + "0000" + str(nm + i) + "_leftImg8bit.png", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, dtype=np.float32)
        image -= mean
        images.append(image)

    label = cv2.imread(root_label + image_name + "000019_gtFine_labelTrainIds.png", cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
    images = torch.from_numpy(np.array(np.transpose(images, (0, 3, 1, 2)), dtype="float32")/255)
    labels = torch.from_numpy(np.array([label], dtype="int64"))
    show_single(args, ColorTransition().recover(labels[0]), "origin")
    inputs = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.int64)
    predict = for_test(args, model, inputs, labels, iou, lstm=args.lstm, need=True)
    print(predict.size())
    color_img = ColorTransition().recover(predict[0])
    show_single(args, color_img, "ls_noise")
    epoch_iou = iou.iou_demo()
    print(epoch_iou)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.use_pre = False
    device = args.device
    model_name = "../saved_model/PSPNet_lstm_noise.pt"
    init_model = load_model(args)
    init_model = convert_model(init_model)
    init_model = init_model.to(device)
    if args.multi_gpu:
        init_model = nn.DataParallel(init_model, device_ids=[0])
    init_model.load_state_dict(torch.load(model_name), strict=True)
    init_model.eval()
    print("load param over")

    # for lstm model inference
    make_image_lstm(init_model, device)

    # # for base model inference only one image as input (root of image is necessary)
    # img_name = "/home/wbw/val/munster/munster_000055_000019_leftImg8bit.png"
    # make_image(init_model, img_name, device)
