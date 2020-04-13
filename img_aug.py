import cv2
import random
import math
from parameter import args


class Aug(object):
    def __init__(self, use_sequence=False):
        self.crop_point = None
        self.flip_point = None
        self.scale_point = None
        self.rotate_point = None
        self.use_scale = args.random_scale
        self.use_mirror = args.random_mirror
        self.use_rotate = args.random_rotate
        self.use_sequence = use_sequence
        self.crop_size = args.input_size
        if self.use_sequence:
            crop_h, crop_w = self.crop_size
            img_h, img_w = args.need_size
            self.scale_point = random.randint(8, 10) / 10.0
            if self.use_scale:
                img_h = math.floor(img_h * self.scale_point)
                img_w = math.floor(img_w * self.scale_point)
            self.crop_point = [random.randint(0, max(0, img_h - crop_h)), random.randint(0, max(0, img_w - crop_w))]
            self.flip_point = random.randint(0, 2)
            self.rotate_point = random.randint(-15, 15)

    def cal(self, image, label):
        if self.use_scale:	 # if use multi scale
            image, label = self.generate_scale_label(image, label)
            image, label = self.scale(image, label, args.ignore_label)  # padding for scale
        image, label = self.crop(image, label)  # random crop
        if self.use_mirror:
            image, label = self.mirror(image, label)
        if self.use_rotate:
            image, label = self.rotate(image, label, args.ignore_label)
        return image, label

    def scale(self, image, label, ignore_label):
        crop_h, crop_w = self.crop_size
        img_h, img_w = label.shape
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:	 # if size after scale smaller than input size padding zero and ignore label
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=ignore_label)
        else:
            img_pad, label_pad = image, label
        return img_pad, label_pad

    def crop(self, image, label):
        crop_h, crop_w = self.crop_size
        img_h, img_w = label.shape	 # 512、1024
        if self.crop_point is None:
            # random crop image
            h_off = random.randint(0, img_h - crop_h)
            w_off = random.randint(0, img_w - crop_w)
        else:
            h_off, w_off = self.crop_point
        image = image[h_off: h_off + crop_h, w_off: w_off + crop_w]
        label = label[h_off: h_off + crop_h, w_off: w_off + crop_w]
        return image, label

    def mirror(self, image, label):
        if self.flip_point is None:
            flip = random.randint(0, 1)
        else:
            flip = self.flip_point
        if flip == 0:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label

    # generate image and label in different scales
    def generate_scale_label(self, image, label):
        if self.scale_point is None:
            f_scale = random.randint(8, 10) / 10.0
        else:
            f_scale = self.scale_point
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def rotate(self, image, label, ignore_label):
        if self.rotate_point is None:
            angle = random.randint(-15, 15)
        else:
            angle = self.rotate_point
        (h, w) = label.shape
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        image = cv2.warpAffine(image, M, borderValue=(0, 0, 0), dsize=(h, w))
        label = cv2.warpAffine(label, M, borderValue=ignore_label, dsize=(h, w))
        return image, label
