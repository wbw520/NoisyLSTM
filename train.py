from model import *
from tools import load_data, load_weight, print_param, fix_parameter
from parameter import args
from train_model import *
import torch.optim as optim
from DANET.danet import get_danet
import torch
from sync_batchnorm import convert_model
from modeling.deeplab import DeepLab
from FCN import FCN8s, VGGNet
import os


def train(name, lstm=False, use_pre=False, noise=False, epoch=20, aux=False, multi_gpu=args.multi):
    dataloaders = load_data(lstm=lstm, random=args.random_sequence, use_noise=noise)
    init_model = PspNet(args.num_classes, use_aux=aux, use_lstm=lstm)
    # init_model = get_danet(args.num_classes, use_lstm=lstm)
    # init_model = DeepLab(backbone='resnet', output_stride=16)

    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # init_model = FCN8s(pretrained_net=vgg_model, n_class=args.num_classes)

    if multi_gpu:
        init_model = convert_model(init_model)

    if use_pre:
        pre_model = PspNet(args.num_classes, use_aux=False, use_lstm=False)
        pre_model.load_state_dict(torch.load("pre_pspnet.pt"), strict=False)
        print("load pre-train param over")
        load_weight(pre_model, init_model)
        # fix_parameter(init_model, ["final3", "final4", "ppm", "CON_ls"])
        # print("param could be trained")
        # print_param(init_model)

    device_ids = [0, 1, 2, 3]
    if multi_gpu:
        print("using multi gpu")
        init_model = init_model.cuda()
        if torch.cuda.device_count() > 1:
            init_model = nn.DataParallel(init_model, device_ids=device_ids)
    else:
        init_model.to(args.gpu)

    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, init_model.parameters()), lr=0.0001)
    # criterion = SoftIoULoss(C.num_class)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    print("start pre train")
    train_model(init_model, dataloaders, criterion, optimizer_ft, name, num_epochs=epoch, use_lstm=lstm, use_aux=aux)


if __name__ == '__main__':
    model_name = "pre_pspnet.pt"
    print(model_name)
    train(model_name, lstm=False, use_pre=False, epoch=50, noise=False, aux=False)