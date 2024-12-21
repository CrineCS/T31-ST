# deep库的导入就一行代码
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2
import re
import numpy as np


def getName(filepath):
    idx = filepath.index('/')
    return [filepath[:idx], filepath[(idx + 1):]]

def get_light_folder(folder_path):
    import os

    # 获取文件夹的名字
    folder_name = os.path.basename(folder_path)

    # 获取文件夹名字的最后两个字符
    folder_name_dic = folder_name.rsplit('_', 2)

    return [folder_name_dic[1], folder_name_dic[2]]

# 自定义排序函数
def sort_key(folder):
    # 使用正则表达式提取数字
    folder_name = folder.name
    match = re.match(r'.*_(\d\.\d)_(\d+)', folder_name)
    if match:
        # 将匹配的字符串转换为浮点数和整数
        multiplier, angle = match.groups()
        return (float(multiplier), int(angle))
    return (0, 0)  # 如果没有匹配，返回一个默认值

def get_full_path():
    folder_path = 'light_data'

    # 检查路径是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 遍历文件夹中的所有项
        path_list = os.scandir(folder_path)
        folders = sorted([item for item in path_list], key=sort_key)
    return folders



if __name__ == "__main__":
    # 模型名
    models_name = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
                   "DeepFace", "DeepID", "ArcFace", "Dlib", 'SFace', "Ensemble"]

    pictureNames = []
    model_name = models_name[6]
    with open('original_data\lfw_funneled\pairs_01.txt', 'r') as f:
        lines = f.readlines()
        n = 0
        for line in lines:
            if line != '\n':
                pictureNames.append(line[:-1])

    folders = get_full_path()
    for entry in folders:
        # 获取完整路径
        resT = 0
        resF = 0
        resC = 0
        res = 0
        # print(entry.path)

        brightness = get_light_folder(entry.path)[0]
        angle = get_light_folder(entry.path)[1]
        print('brightness:' + brightness + ' ;angle:' + angle)

        for i in range(300):
            complete = []
            for j in range(4):
                names = getName(pictureNames[4 * i + j])

                complete.append(entry.path + "\\" + names[0] + "\\" + names[1])


            result_true = DeepFace.verify(img1_path=complete[0],
                                          img2_path=complete[1],
                                          model_name=model_name,
                                          enforce_detection=False)

            if result_true['verified'] is True:
                resT += 1
                resC += 1

            result_false = DeepFace.verify(img1_path=complete[2],
                                           img2_path=complete[3],
                                           model_name=model_name,
                                           enforce_detection=False)

            if result_false['verified'] is False:
                resF += 1
                resC += 1

        res = round(resC/600, 2)


        with open('record.txt', 'a') as file:
            # 追加内容
            file.write('{}_{}: {} t: {}/300 f: {}/300\n'.format(brightness, angle, res, resT, resF))


