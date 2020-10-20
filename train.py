from tools.tool import ICNetLoss, load_model
from engine import *
from tools.data_gen import load_data
import torch.optim as optim
import torch
import torch.nn as nn
from tools.sync_batchnorm import convert_model
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--model_name", type=str, default="PSPNet")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--sequence_len", type=int, default=4,
                        help="Length of sequence for LSTM model.")
    parser.add_argument("--data_dir", type=str, default="/home/wangbowen/PycharmProjects/city_data2/",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--data_extra", type=str, default="/home/wangbowen/PycharmProjects/data_eye_train/",
                        help="Path to the directory of noise data")
    parser.add_argument("--original_size", type=int, default=[1024, 2048],
                        help="original size of data set image.")
    parser.add_argument("--need_size", type=int, default=[512, 1024],
                        help="image size require for this program.")
    parser.add_argument("--input_size", type=int, default=[448, 448],
                        help="Comma-separated string with height and width of images. It also consider as crop size.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--ignore_label", type=int, default=19,
                        help="this kind of pixel will not used for both train and evaluation")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_epoch", type=int, default=40,
                        help="Number of training steps.")
    parser.add_argument("--use_pre", type=bool, default=False)

    # lstm setting
    parser.add_argument("--lstm", type=bool, default=False)
    parser.add_argument("--merge", type=bool, default=True,
                        help="merge previous frame, set false for all frame visualization")
    parser.add_argument("--lstm_kernel", type=int, default=1)
    parser.add_argument("--lstm_layer", type=int, default=1)
    parser.add_argument("--noise", type=bool, default=False)
    parser.add_argument("--frame_cut", type=int, default=1)
    parser.add_argument("--noise_type", type=str, default="extra")

    # augment tools
    parser.add_argument("--random_crop", type=bool, default=True,
                        help="Whether to randomly crop the inputs during the training.")
    parser.add_argument("--random_mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_rotate", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--noise_ratio", type=int, default=50,
                        help="define the possibility of noise.")

    # evaluation settings
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--show_sequence', default=False, type=bool)
    parser.add_argument('--multi_gpu', default=False, type=bool)
    parser.add_argument("--device", type=str, default='cuda',
                        help="choose gpu device.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def train(args):
    dataloaders = load_data(args, lstm=args.lstm, use_noise=args.noise)
    device = torch.device(args.device)
    init_model = load_model(args)

    if args.multi_gpu:
        device_ids = [0, 1]
        init_model = convert_model(init_model)
        init_model = init_model.cuda()
        if torch.cuda.device_count() > 1:
            init_model = nn.DataParallel(init_model, device_ids=device_ids)
    else:
        init_model.to(device)

    n_parameters = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, init_model.parameters()), lr=args.learning_rate)
    if args.model_name == "ICNet":
        criterion = ICNetLoss(ignore_index=args.ignore_label).to(args.device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    print("start train")
    save_name = args.model_name + f"{'_lstm' if args.lstm else ''}" + f"{'_noise' if args.noise else ''}"
    print("model name: ", save_name)
    train_model(args, init_model, dataloaders, criterion, optimizer,
                save_name, num_epochs=args.num_epoch, use_lstm=args.lstm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)