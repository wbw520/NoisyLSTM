from model import *
from tools import load_data, load_weight, print_param, fix_parameter
import prepare_things as prt
from train_model import *
import torch.optim as optim
from DANET.danet import get_danet
import torch
from sync_batchnorm import convert_model
from modeling.deeplab import DeepLab
from FCN import FCN8s, VGGNet
import os
from models.icnet import ICNet
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--model_name", type=str, default="PSPnet")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--sequence_len", type=int, default=4,
                        help="Length of sequence for LSTM model.")
    parser.add_argument("--data-dir", type=str, default="/home/wangbowen/PycharmProjects/city_data2/",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--data-extra", type=str, default="/home/wangbowen/PycharmProjects/data_eye_train/",
                        help="Path to the directory of noise data")
    parser.add_argument("--original-size", type=int, default=[1024, 2048],
                        help="original size of data set image.")
    parser.add_argument("--need-size", type=int, default=[512, 1024],
                        help="image size require for this program.")
    parser.add_argument("--input-size", type=int, default=[448, 448],
                        help="Comma-separated string with height and width of images. It also consider as crop size.")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--ignore-label", type=int, default=19,
                        help="this kind of pixel will not used for both train and evaluation")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epoch", type=int, default=40,
                        help="Number of training steps.")
    parser.add_argument("--seq", type=bool, default=False,
                        help="whether use LSTM model")
    parser.add_argument("--aux", type=bool, default=True,
                        help="whether use aux branch for training")

    # augment tools
    parser.add_argument("--random-crop", type=bool, default=True,
                        help="Whether to randomly crop the inputs during the training.")
    parser.add_argument("--random-mirror", type=bool, default=False,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-rotate", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-sequence", type=bool, default=False,
                        help="Whether to random sequence.")
    parser.add_argument("--noise-ratio", type=int, default=0,
                        help="define the possibility of noise.")

    # evaluation settings
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--show_sequence', default=True, type=str)
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument("--local_rank", type=int)
    # parser.add_argument("--snapshot-dir", type=str, default="PSP.pt",
    #                     help="name to save the model.")
    parser.add_argument('--multi_gpu', default=True, type=str)
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="choose gpu device.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def train(args, name, lstm=False, use_pre=False, noise=False, epoch=20, aux=False):
    # prt.init_distributed_mode(args)
    dataloaders = load_data(args, lstm=lstm, random=args.random_sequence, use_noise=noise)
    device = torch.device(args.device)
    init_model = PspNet(args, use_aux=aux, use_lstm=lstm)
    # init_model = ICNet(args, use_lstm=lstm)
    # init_model = get_danet(args.num_classes, use_lstm=lstm)
    # init_model = DeepLab(backbone='resnet', output_stride=16)

    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # init_model = FCN8s(pretrained_net=vgg_model, n_class=args.num_classes)

    if use_pre:
        if args.model_name == "ICnet":
            pre_model = ICNet(args, use_lstm=False)
        else:
            pre_model = PspNet(args, use_aux=False, use_lstm=False)
        pre_model.load_state_dict(torch.load("saved_model/ICnet.pt"), strict=False)
        print("load pre-train param over")
        load_weight(pre_model, init_model)
        # fix_parameter(init_model, ["final1", "final2", "CON_ls"])
        # print("param could be trained")
        # print_param(init_model)

    device_ids = [0, 1]
    if args.multi_gpu:
        print("using multi gpu")
        init_model = convert_model(init_model)
        init_model = init_model.cuda()
        if torch.cuda.device_count() > 1:
            init_model = nn.DataParallel(init_model, device_ids=device_ids)
    else:
        init_model.to(device)

    n_parameters = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, init_model.parameters()), lr=0.0001)
    if args.model_name == "ICnet":
        criterion = ICNetLoss(ignore_index=args.ignore_label).to(args.device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    print("start train")
    print(args.model_name)
    train_model(args, init_model, dataloaders, criterion, optimizer, name, num_epochs=epoch, use_lstm=lstm, use_aux=aux)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = "LLS_"
    train(args, args.model_name+model_name, lstm=True, use_pre=True, epoch=40, noise=False, aux=False)