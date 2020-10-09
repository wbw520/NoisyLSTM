import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np


def show_single(image):
    # show single image
    image = cv2.resize(image, (1280, 1024), interpolation=cv2.INTER_NEAREST)
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    plt.imshow(image)
    plt.axis('on')
    plt.show()


class ImageAugment(object):
    """
    class for augment the training data using imgaug
    """
    def __init__(self, args):
        self.args = args
        self.key = 0
        self.rotate = np.random.randint(-15, 15)
        self.scale_x = random.uniform(0.8, 1.2)
        self.scale_y = random.uniform(0.8, 1.2)
        self.translate_x = random.uniform(-0.2, 0.2)
        self.translate_y = random.uniform(-0.2, 0.2)
        self.brightness = np.random.randint(-10, 10)
        self.linear_contrast = random.uniform(0.5, 2.0)
        self.alpha = random.uniform(0, 1.0)
        self.lightness = random.uniform(0.75, 1.5)
        self.Gaussian = random.uniform(0.0, 0.05*255)
        self.Gaussian_blur = random.uniform(0, 3.0)
        self.crop_percent = random.uniform(0, 0.1)
        self.cval = np.random.randint(0, 255)
        self.AverageBlur = random.randint(2, 7)
        self.MedianBlur = random.randint(3, 11)
        if self.MedianBlur % 2 == 0:
            self.MedianBlur += 1
        self.LinearContrast_per_channel = bool(random.getrandbits(1))
        self.Add_per_channel = bool(random.getrandbits(1))
        self.AdditiveGaussianNoise_per_channel = bool(random.getrandbits(1))

        self.switches = [False for i in range(9)]
        for i in range(random.randint(1, 5)):
            switch_id = random.randint(0, len(self.switches)-1)
            while self.switches[switch_id]:
                switch_id = random.randint(0, len(self.switches)-1)
            self.switches[switch_id] = True

        self.HueAndSaturation_value = random.randint(-20, 20)
        self.LinearContrast_value = random.uniform(0.5, 2.0)
        self.LinearContrast_per_channel = bool(random.getrandbits(1))

    def aug(self, image, label, sequence, color=True):
        """
        :param image: need size (H, W, C) one image once
        :param label: need size same as image or (H, W)(later translate to size(1, H, W, C))
        :param sequence: collection of augment function
        :return:
        """
        if color:
            label = np.expand_dims(label, axis=-1)
        image_aug, label_aug = sequence(image=image, segmentation_maps=np.expand_dims(label, axis=0))
        if color:
            label_aug = np.squeeze(label_aug, axis=-1)
        label_aug = np.squeeze(label_aug, axis=0)
        return image_aug, label_aug

    def rd(self, hehe):
        seed = np.random.randint(0, hehe)
        return seed

    def aug_sequence(self):
        sequence = self.aug_function()
        seq = iaa.Sequential(sequence, random_order=False)
        return seq

    def aug_function(self):
        sequence = []
        if self.rd(2) == self.key:
            sequence.append(iaa.Fliplr(1.0))  # 50% horizontally flip all batch images
        # if self.rd(2) == self.key:
        #     sequence.append(iaa.Flipud(1.0))  # 50% vertically flip all batch images

        #### Safe Augmentation
        # sequence.append(iaa.Crop(
        #     percent=self.crop_percent,
        # ))
        sequence.append(iaa.Affine(
            scale={"x": self.scale_x, "y": self.scale_y},  # scale images to 80-120% of their size
            translate_percent={"x": self.translate_x, "y": self.translate_y},  # translate by -20 to +20 percent (per axis)
            rotate=(self.rotate),  # rotate by -15 to +15 degrees
            cval=self.args.ignore_label
        ))

        # ##### Dangerous Augmentation
        # if self.switches[0]:
        #     sequence.append(
        #         iaa.GaussianBlur(self.Gaussian_blur),  # blur images with a sigma between 0 and 3.0
        #     )
        # if self.switches[1]:
        #     sequence.append(
        #         iaa.AverageBlur(self.AverageBlur),  # blur images using local means with kernel size 2-7
        #     )
        # if self.switches[2]:
        #     sequence.append(
        #         iaa.MedianBlur(self.MedianBlur)  # blur images using local medians with kernel size 3-11
        #     )
        # if self.switches[3]:
        #     sequence.append(
        #         iaa.Sharpen(alpha=self.alpha, lightness=self.lightness),  # sharpen images
        #     )
        # if self.switches[4]:
        #     sequence.append(
        #         iaa.LinearContrast(self.linear_contrast, per_channel=self.LinearContrast_per_channel),  # improve or worse the contrast
        #     )
        # if self.switches[5]:
        #     sequence.append(
        #         iaa.Add(self.brightness, per_channel=self.Add_per_channel),  # change brightness
        #     )
        # if self.switches[6]:
        #     sequence.append(
        #         iaa.AdditiveGaussianNoise(loc=0, scale=0.1, per_channel=self.AdditiveGaussianNoise_per_channel)  # add gaussian n
        #     )
        # if self.switches[7]:
        #     sequence.append(
        #         iaa.AddToHueAndSaturation(self.HueAndSaturation_value), # change hue and saturation
        #     )
        # if self.switches[8]:
        #     sequence.append(
        #         iaa.LinearContrast(self.LinearContrast_value, per_channel=self.LinearContrast_per_channel), # improve or worsen the contrast
        #     )

        return sequence


def show_aug(image, label):
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    for i in range(1, len(image)+1):
        plt.subplot(len(image), 2, 2*i-1)
        plt.imshow(image[i-1])
        plt.subplot(len(image), 2, 2*i)
        plt.imshow(label[i-1])
    plt.show()


"""
see the data augment
"""

# wbw = ImageAugment()
# seq = wbw.aug_sequence()
# name_list = ["003"]
# for i in name_list:
#     label = cv2.imread(C.data_root + "seq_13/labels/frame"+i+".png")  # , cv2.IMREAD_GRAYSCALE
#     label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#     label = cv2.resize(label, (448, 448), interpolation=cv2.INTER_NEAREST)
#     image = cv2.imread(C.data_root + "seq_13/left_frames/frame"+i+".png")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_NEAREST)
#     image_aug, label_aug = wbw.aug(image, label, seq, color=False)
#     show_single(image)
#     show_single(image_aug)
