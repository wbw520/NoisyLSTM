from model import *
from tools import load_data, load_weight, print_param
from parameter import args
from train_model import *
import torch.optim as optim
import torch


def train(name, lstm=False, use_pre=False, noise=False, epoch=20):
    dataloaders = load_data(lstm=lstm, random=args.random_sequence, use_noise=noise)
    if lstm:
        aux = False
    else:
        aux = True
    init_model = PspNet(args.num_classes, use_aux=aux, use_lstm=lstm)
    # vgg_model = VGGNet(requires_grad=True, shnow_params=False)
    # init_model = FCN8s(pretrained_net=vgg_model, n_class=12)

    if use_pre:
        pre_model = PspNet(args.num_classes, use_aux=False, use_lstm=False)
        pre_model.load_state_dict(torch.load("psp3.pt"), strict=False)
        print("load pre-train param over")
        load_weight(pre_model, init_model)
        # fix_parameter(init_model, ["final3", "final4", "ppm"])
        # print("param could be trained")
        # print_param(init_model)

    init_model.to(args.gpu)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, init_model.parameters()), lr=0.0001)
    # criterion = SoftIoULoss(C.num_class)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    print("start pre train")
    train_model(init_model, dataloaders, criterion, optimizer_ft, name, num_epochs=epoch, use_lstm=lstm)


if __name__ == '__main__':
    model_name = "ls_3.pt"
    print(model_name)
    train(model_name, lstm=True, use_pre=True, epoch=40, noise=False)