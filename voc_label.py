import pickle
import os
from PIL import Image

photo_dir_path = r"F:\word and num\Text localization\Challenge2_Training_Task12_Images"
text_dir_path = r"F:\word and num\Text localization\Challenge2_Training_Task1_GT"
aim_dir = r"E:\2021homework\yolo\data\mydata\labels\train"
classes = '0'


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = -(box[0] - box[2])
    h = -(box[1] - box[3])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


# 先读图片，再导入到对应的txt文件中
def convert_dir(photo_dir_path, text_dir_path, aim_dir):
    f_dir = os.listdir(photo_dir_path)
    for each in f_dir:
        img = Image.open(photo_dir_path + r"\\" + each)
        sz = img.size
        text_name = "gt_" + each.strip("jpg") + "txt"
        with open(text_dir_path + '\\' + text_name, "r") as fr:
            text = fr.read()
            text_loc = [int(num) for num in text.split(' ')[:4]]
            # print(text_loc)
            text_loc_after = convert(sz, text_loc)
            # print(text_loc_after)
            fr.close()
        text_name = each.strip("jpg") + "txt"
        with open(aim_dir + '\\' + text_name, "w") as fw:
            fw.write(classes + " ")
            m = 0
            for loc in text_loc_after:
                if m == 3:
                    fw.write(str(loc))
                else:
                    fw.write(str(loc) + " ")
                m += 1


convert_dir(photo_dir_path, text_dir_path, aim_dir)
