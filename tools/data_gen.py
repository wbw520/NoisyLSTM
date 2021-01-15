import numpy as np
from random import randint
import cv2
import os
import torch
from tools.img_aug import Aug
from tools.tool import ColorTransition
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader


class DataSet(object):
    def __init__(self, args, data_list, train=True, use_aug=True):
        self.data_list = data_list
        self.train = train
        self.use_aug = use_aug
        self.args = args
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    # return one sample image and its label
    def __getitem__(self, index):
        datafiles = self.data_list[index]
        image, label = get_image(self.args, datafiles)
        image = np.array(image, dtype=np.float32)
        image -= self.mean	 # sub mean
        if self.train and self.use_aug:
            image, label = Aug(self.args).cal(image, label)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image/255)  # convert numpy data to tensor
        label = torch.from_numpy(label)
        return {"image": image, "label": label, "name": datafiles[0]}


class DataSetSequence(object):
    def __init__(self, args, data_list, train=True, use_aug=True, use_noise=False):
        self.data_list = data_list
        self.use_noise = use_noise
        self.train = train
        self.use_aug = use_aug
        self.args = args
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    # return one sample image and its label
    def __getitem__(self, index):
        sequence_data = self.data_list[index]
        sequence_len = len(sequence_data)
        images = []
        labels = []
        names = []
        limit = sequence_len//2
        if self.train and self.use_aug:
            wbw = Aug(self.args, use_sequence=True)
        for i in range(sequence_len):
            others = False
            if self.use_noise and limit > 0 and i != sequence_len-1 and self.train:
                if randint(0, 100) > self.args.noise_ratio:
                    limit -= 1
                    if self.args.noise_type == "extra":
                        sequence_data[i][0] = noise(self.args)
                    else:
                        others = True
            image, label = get_image(self.args, sequence_data[i])
            image = np.array(image, dtype=np.float32)
            if others:
                if self.args.noise_type == "distortion":
                    image = niuqu(image)
                elif self.args.noise_type == "gauss":
                    image = gauss(image)
                elif self.args.noise_type == "random":
                    h, w, c = image.shape
                    image = np.random.uniform(0, 255, (h, w, c))
                else:
                    raise Exception("unexpected noise type")
            image -= self.mean	 # sub mean
            if self.train and self.use_aug:
                image, label = wbw.cal(image, label)
            images.append(image)
            labels.append(label)
            names.append(sequence_data[i][0])

        images = torch.from_numpy(np.array(np.transpose(images, (0, 3, 1, 2)), dtype="float32")/255)
        labels = torch.from_numpy(np.array(labels, dtype="int64"))
        return {"image": images, "label": labels, "name": names}


def get_image(args, datafiles):
    image = cv2.imread(datafiles[0], cv2.IMREAD_COLOR)	 # shape(1024,2048,3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(datafiles[1], cv2.IMREAD_GRAYSCALE)	 # shape(1024,2048)
    label = ColorTransition().id2trainIdbinary(label)
    need_h, need_w = args.need_size   # resize image for crop larger view area
    image = cv2.resize(image, (need_w, need_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (need_w, need_h), interpolation=cv2.INTER_NEAREST)
    return image, label


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the name of each frame
    """
    def __init__(self, args):
        self.root = "data/"
        self.args = args

    def make_list(self):
        train_list = self.make_list_unit("train")
        val_list = self.make_list_unit("val")
        return {"train": train_list, "val": val_list}

    def make_list_unit(self, mode):
        all_file = self.read_txt(self.root + mode)
        return all_file

    def read_txt(self, root):  # get image and label root for one city
        train_name = []
        label_name = []
        with open(root + "Images.txt", 'r', encoding='UTF-8') as data:
            lines = data.readlines()
            for i in range(len(lines)):
                train_name.append(lines[i][:-1])

        with open(root + "Labels.txt", 'r', encoding='UTF-8') as data:
            lines = data.readlines()
            for i in range(len(lines)):
                label_name.append(lines[i][:-1])

        total = []
        for i in range(len(train_name)-1):
            total.append([self.args.data_dir + train_name[i], self.args.data_dir + label_name[i]])

        return total


class MakeListSequence(object):
    """
    this class used to make list of data for model train and test, return the name of each frame
    """
    def __init__(self, args, batch, random=False):
        self.root = "data/"
        self.batch_size = batch
        self.random = random
        self.frame_cut = args.frame_cut
        self.args = args

    def make_list(self):
        train_list = self.make_list_unit("train", for_train=self.random)
        val_list = self.make_list_unit("val")
        return {"train": train_list, "val": val_list}

    def make_list_unit(self, mode, for_train=False):
        all_file = self.read_txt(self.root + mode, mode, for_train)
        return all_file

    def read_txt(self, root, mode, for_train):  # get image and label root for one city
        train_name = []
        label_name = []
        with open(root + "Images.txt", 'r', encoding='UTF-8') as data:
            lines = data.readlines()
            for line in lines:
                train_name.append(line[:-1])

        with open(root + "Labels.txt", 'r', encoding='UTF-8') as data:
            lines = data.readlines()
            for line in lines:
                label_name.append(line[:-1])

        total = []
        for i in range(len(train_name)):
            temp = []
            elements = train_name[i].split("/")
            folder = elements[2]
            number = elements[3].split("_")
            frame_no = self.drop(number[2])
            image_root = self.args.data_dir + "leftImg8bit/" + mode + "/" + folder + "/"
            for j in range(self.args.sequence_len):
                current_no = frame_no - (self.args.sequence_len - j - 1)*self.frame_cut
                current_no_str = self.name_translation(current_no)
                current_img_name = number[0] + "_" + number[1] + "_" + current_no_str + "_" + "leftImg8bit.png"
                frame = [image_root + current_img_name, self.args.data_dir + label_name[i]]
                temp.append(frame)
            total.append(temp)
        return total

    def drop(self, number):
        for i in range(len(number)):
            if number[i] != "0":
                return int(number[i:])

    def name_translation(self, name):
        # used to translate name in to "000" structure
        name = str(name)
        len_zero = 6 - len(name)
        final = len_zero*"0" + name
        return final


def noise(args):
    ff = ["left_frames", "right_frames"]
    folder_name = get_name(args.data_extra)
    selected_id_folder = np.random.randint(0, len(folder_name))
    selected_folder_root = args.data_extra + folder_name[selected_id_folder] + "/" + ff[np.random.randint(0, 2)] + "/"
    images = get_name(selected_folder_root, mode_folder=False)
    selected_id_image = np.random.randint(0, len(images))
    while "png" not in images[selected_id_image]:
        selected_id_image = np.random.randint(0, len(images))
    selected_image_name = selected_folder_root + images[selected_id_image]
    return selected_image_name


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def niuqu(image):
    seq = iaa.Sequential([iaa.PiecewiseAffine(scale=0.2, seed=1)])
    im = seq(image=image)
    return im


def gauss(image):
    seq = iaa.Sequential([iaa.GaussianBlur(10, seed=1)])
    im = seq(image=image)
    return im


def load_data(args, lstm, use_noise=False):
    shuffle = {"train": True, "val": False}
    if lstm:
        L = MakeListSequence(args, args.data_dir, args.sequence_len).make_list()
        sequence_dataset = {"train": DataSetSequence(args, L["train"], use_noise=use_noise),
                            "val": DataSetSequence(args, L["val"], train=False, use_aug=False, use_noise=False)}
        dataloaders_lstm = {x: DataLoader(sequence_dataset[x], batch_size=args.batch_size//args.sequence_len, shuffle=shuffle[x], num_workers=4)
                            for x in ["train", "val"]}
        print("load lstm data over")
        return dataloaders_lstm
    else:
        L = MakeList(args).make_list()
        image_dataset = {"train": DataSet(args, L["train"]),
                         "val": DataSet(args, L["val"], train=False, use_aug=False)}
        dataloaders = {x: DataLoader(image_dataset[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                       for x in ["train", "val"]}
        print("load normal data over")
        return dataloaders

# """
# see the sequence data generation
# """
# L = MakeListSequence(args.data_dir, 4, random=True).make_list()
# print(L["train"][0])
# dataloaders = DataSetSequence(L["train"], use_noise=True)
# for i_batch, sample_batch in enumerate(dataloaders):
#     print(sample_batch["image"].size())