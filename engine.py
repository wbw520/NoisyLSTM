import time
import torch
from tools.tool import IouCal, predict_sliding
from tqdm.auto import tqdm


def train_model(args, my_model, dataloaders, criterion, optimizer, save_name, num_epochs=40, use_lstm=False, use_aux=False):
    start_time = time.time()
    device = args.device
    train_miou_history = []
    val_miou_history = []
    best_iou = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)
        # for every epoch there is train and val phase respectively
        for phase in ["train", "val"]:
            iou = IouCal(args)
            if phase == "train":
                print("start_training round" + str(epoch))
                print(optimizer.param_groups[0]["lr"])
                my_model.train()  # set model to train
            else:
                print("start_val round" + str(epoch))
                my_model.eval()   # set model to evaluation

            running_loss = 0.0
            for i_batch, sample_batch in enumerate(tqdm(dataloaders[phase])):
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

                # zero the gradient parameter
                optimizer.zero_grad()
                if phase == "train":
                    a = for_train(my_model, inputs, labels, optimizer, criterion, iou, use_aux=use_aux)
                    running_loss += a
                elif not args.random_crop:
                    for_val(my_model, inputs, labels, iou)
                else:
                    for_test(args, my_model, inputs, labels, iou, lstm=use_lstm)

            epoch_iou = iou.iou_demo()
            epoch_loss = round(running_loss / len(dataloaders[phase]), 3)
            if phase == "train":
                train_miou_history.append(epoch_iou)
                print("{} Loss: {:.4f} iou: {:.4f}".format(phase, epoch_loss, epoch_iou))

            if phase == "val":
                if epoch == 20:
                    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                if epoch_iou > best_iou:
                    best_iou = epoch_iou
                    torch.save(my_model.state_dict(), "saved_model/" + save_name + ".pt")
                    print("get higher iou save current model")
                val_miou_history.append(epoch_iou)
                print('val miou history: ', val_miou_history)
                print('train miou history: ', train_miou_history)

    time_elapsed = time.time() - start_time
    print("training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


def for_train(my_model, inputs, labels, optimizer, criterion, iou, use_aux):
    # forward
    # track history if only in train
    with torch.set_grad_enabled(True):
        if use_aux:         # not use aux when lstm mode
            outputs, aux_outputs = my_model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
        else:
            outputs = my_model(inputs)
            loss = criterion(outputs, labels)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    _, preds = torch.max(outputs, 1)   # (H, W)
    iou.evaluate(labels, preds)
    loss.backward()
    optimizer.step()

    # statistics
    a = loss.item()
    return a


def for_val(my_model, inputs, labels, iou, need=False):
    with torch.set_grad_enabled(False):
        outputs = my_model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    _, preds = torch.max(outputs, 1)   # (H, W)
    iou.evaluate(labels, preds)
    if need:
        return preds


def for_test(args, my_model, inputs, labels, iou, lstm, need=False):
    pred = predict_sliding(args, my_model, inputs, args.input_size, args.num_classes, lstm)
    if iou:
        iou.evaluate(pred, labels)
    if need:
        return pred