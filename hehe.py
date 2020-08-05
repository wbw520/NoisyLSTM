from img_aug import Aug
import cv2
from tools import show_single

wbw =Aug(use_sequence=True)
name_list = [["leftImg8bit/train/aachen/aachen_000000_000007_leftImg8bit.png", "gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"],
            ["leftImg8bit/train/aachen/aachen_000000_000013_leftImg8bit.png", "gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"],
             ["leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png", "gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"]]

for i in range(len(name_list)):
    image = cv2.imread("/home/wangbowen/PycharmProjects/city_data/" + name_list[i][0])
    label = cv2.imread("/home/wangbowen/PycharmProjects/city_data/" + name_list[i][1])
    image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, label = wbw.cal(image, label)
    print(name_list[i][0])
    show_single(image)
