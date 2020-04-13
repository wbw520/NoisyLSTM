import numpy as np
from random import randint
import cv2
import os
import torch
from img_aug import Aug
from augg import ImageAugment
from parameter import args


class DataSet(object):
    def __init__(self, data_list, train=True, use_aug=True):
        self.data_list = data_list
        self.train = train
        self.use_aug = use_aug
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    # return one sample image and its label
    def __getitem__(self, index):
        datafiles = self.data_list[index]
        image, label = get_image(datafiles)
        # image = np.array(image, dtype=np.float32)
        # image -= self.mean	 # sub mean
        if self.train and self.use_aug:
            wbw = ImageAugment()
            seq = wbw.aug_sequence()
            image, label = wbw.aug(image, label, seq)
            # image, label = Aug().cal(image, label)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image/255)  # convert numpy data to tensor
        label = torch.from_numpy(label)
        return {"image": image, "label": label, "name": datafiles[0]}


class DataSetSequence(object):
    def __init__(self, data_list, train=True, use_aug=True, use_noise=False):
        self.data_list = data_list
        self.use_noise = use_noise
        self.train = train
        self.use_aug = use_aug
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
            wbw = ImageAugment()
            seq = wbw.aug_sequence()
        for i in range(sequence_len):
            if self.train and self.use_noise and limit > 0 and i != sequence_len-1:
                if randint(0, 100) > args.noise_ratio:
                    limit -= 1
                    sequence_data[i][0] = noise()
            image, label = get_image(sequence_data[i])
            # image = np.array(image, dtype=np.float32)
            # image -= self.mean	 # sub mean
            if self.train and self.use_aug:
                image, label = wbw.aug(image, label, seq)
            images.append(image)
            labels.append(label)
            names.append(sequence_data[i][0])

        images = torch.from_numpy(np.array(np.transpose(images, (0, 3, 1, 2)), dtype="float32")/255)
        labels = torch.from_numpy(np.array(labels, dtype="int64"))
        return {"image": images, "label": labels, "name": names}


def get_image(datafiles):
    image = cv2.imread(datafiles[0], cv2.IMREAD_COLOR)	 # shape(1024,2048,3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(datafiles[1], cv2.IMREAD_GRAYSCALE)	 # shape(1024,2048)
    need_h, need_w = args.need_size   # resize image for crop larger view area
    image = cv2.resize(image, (need_w, need_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (need_w, need_h), interpolation=cv2.INTER_NEAREST)
    return image, label


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the name of each frame
    """
    def __init__(self, root):
        self.root = root + "list/"

    def make_list(self):
        train_list = self.make_list_unit("train")
        val_list = self.make_list_unit("val")
        return {"train": train_list, "val": val_list}

    def make_list_unit(self, mode):
        all_file = []
        folder_name = get_name(self.root + mode + "/")
        for i in folder_name:
            current_city_root = self.root + mode + "/" + i + "/"
            current_city_inf = self.read_txt(current_city_root)
            all_file.extend(current_city_inf)
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
            total.append([args.data_dir + train_name[i], args.data_dir + label_name[i]])

        return total


class MakeListSequence(object):
    """
    this class used to make list of data for model train and test, return the name of each frame
    """
    def __init__(self, root, batch, random=False):
        self.root = root + "list/"
        self.batch_size = batch
        self.random = random

    def make_list(self):
        train_list = self.make_list_unit("train", for_train=self.random)
        val_list = self.make_list_unit("val")
        return {"train": train_list, "val": val_list}

    def make_list_unit(self, mode, for_train=False):
        all_file = []
        folder_name = get_name(self.root + mode + "/")
        for i in folder_name:
            current_city_root = self.root + mode + "/" + i + "/"
            current_city_inf = self.read_txt(current_city_root, for_train)
            all_file.extend(current_city_inf)
        return all_file

    def read_txt(self, root, for_train):  # get image and label root for one city
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
        for i in range(len(train_name)-1):
            temp = []
            for j in range(self.batch_size)[::-1]:
                if i < j:
                    frame_id = 0
                else:
                    frame_id = i - j

                if for_train and j > 1 and randint(0, 100) > args.noise_ratio and i > 0:
                    frame_cu = frame_id + randint(-2, 2)
                    while not (frame_cu >= 0 and frame_cu < i):
                        frame_cu = frame_id + randint(-2, 2)
                    frame_id = frame_cu

                frame = [args.data_dir + train_name[frame_id], args.data_dir + label_name[frame_id]]
                temp.append(frame)
            total.append(temp)
        return total


def noise():
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


# """
# see the sequence data generation
# """
# L = MakeListSequence(args.data_dir, 4, random=True).make_list()
# dataloaders = DataSetSequence(L["train"], use_noise=True)
# for i_batch, sample_batch in enumerate(dataloaders):
#     print(sample_batch["image"].size())