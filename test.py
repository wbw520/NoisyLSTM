from tools import load_data, ColorTransition, predict_sliding, IouCal, show_single
from model import PspNet
from train_model import for_test
import torch
import argparse
import torch.nn as nn
from models.icnet import ICNet
from sync_batchnorm import convert_model
from train import get_args_parser
from tqdm.auto import tqdm


def test_model(model, lstm, noise, device):
    dataloaders = load_data(args, lstm=lstm, random=args.random_sequence, use_noise=noise)
    iou = IouCal(args)
    for i_batch, sample_batch in enumerate(tqdm(dataloaders["val"])):
        if len(list(sample_batch["image"])) < args.batch_size//args.sequence_len:
            continue
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        if use_lstm:
            inputs = torch.cat(list(inputs), dim=0)
            labels = list(labels)
            # spilt final frame label for each sequence
            label_for_pred = []
            for i in range(len(labels)):
                label_for_pred.append(labels[i][-1:])
            labels = torch.cat(label_for_pred, dim=0)
        predict = for_test(args, model, inputs, labels, iou, lstm=use_lstm)
        # color_img = ColorTransition().recover(torch.squeeze(predict, dim=0))
        # show_single(args, color_img)
    epoch_iou = iou.iou_demo()
    print(epoch_iou)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = "PSPnetLLS_.pt"
    use_lstm = True
    init_model = PspNet(args, use_aux=False, use_lstm=use_lstm)
    # init_model = ICNet(args, use_lstm=use_lstm)
    device_ids = [0]
    if args.multi_gpu:
        print("using multi gpu")
        init_model = convert_model(init_model)
        init_model = init_model.cuda()
        if torch.cuda.device_count() > 1:
            init_model = nn.DataParallel(init_model, device_ids=device_ids)
    else:
        init_model.to(args.device)
    init_model.load_state_dict(torch.load("saved_model/" + model_name, map_location=args.device), strict=True)
    init_model.eval()
    print("load pre-train param over")
    test_model(init_model, lstm=use_lstm, noise=True, device=args.device)