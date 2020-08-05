from tools import load_data, ColorTransition, predict_sliding, IouCal, show_single
from parameter import args
from model import PspNet
from train_model import for_val
import torch
from tqdm.auto import tqdm


def test_model(model, lstm, noise, device):
    dataloaders = load_data(lstm=lstm, random=args.random_sequence, use_noise=noise, batch_train=1)
    iou = IouCal()
    for i_batch, sample_batch in enumerate(tqdm(dataloaders["val"])):
        if use_lstm:
            inputs = list(sample_batch["image"])
            if len(inputs) != 1:
                inputs = torch.cat(inputs, dim=0).to(device, dtype=torch.float32)
            else:
                inputs = inputs[0].to(device, dtype=torch.float32)
            labels = list(sample_batch["label"])
            # spilt final frame label for each sequence
            label_for_pred = []
            for i in range(len(labels)):
                label_for_pred.append(labels[i][-1:])
            if len(label_for_pred) != 1:
                labels = torch.cat(label_for_pred, dim=0).to(device, dtype=torch.int64)
            else:
                labels = label_for_pred[0].to(device, dtype=torch.int64)
        else:
            inputs = sample_batch["image"].to(device, dtype=torch.float32)
            labels = sample_batch["label"].to(device, dtype=torch.int64)
        predict = for_val(model, inputs, labels, iou, need=True)
        color_img = ColorTransition().recover(torch.squeeze(predict, dim=0))
        show_single(color_img)

        iou.evaluate(predict, labels)
        print(sample_batch["name"][-1])
    epoch_iou = iou.iou_demo()
    print(epoch_iou)


if __name__ == '__main__':
    model_name = "psp_full_3.pt"
    use_lstm = False
    init_model = PspNet(args.num_classes, use_aux=False, use_lstm=use_lstm)
    init_model.load_state_dict(torch.load(model_name, map_location=args.gpu), strict=False)
    init_model.to(args.gpu)
    init_model.eval()
    print("load pre-train param over")
    test_model(init_model, lstm=use_lstm, noise=False, device=args.gpu)