import face_recognition
import cv2
import numpy as np
import os

pictureNames=[]

def getName(filepath):
    idx=filepath.index('/')
    return[filepath[:idx],filepath[(idx+1):]]

with open('Test/lfw/pairs_01.txt','r') as f:
    lines=f.readlines()
    n=0
    for line in lines:
        if line !='\n':                
            pictureNames.append(line[:-1])
    
    for i in range(300):
        complete=[]
        for j in range(4):
            names=getName(pictureNames[4*i+j])
            complete.append("Test\lfw\\"+names[0]+"\\"+names[1])

for one_path in pictureNames:
    print(one_path)
    names=getName(one_path)
    # 加载图片
    image_path="Test\lfw\\"+names[0]+"\\"+names[1]
    frame = cv2.imread(image_path)

    # 缩小帧尺寸，提高处理速度
    small_frame = cv2.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4)
    # 转换为 RGB 格式
    rgb_frame = small_frame[:, :, ::-1]

    # 检测人脸关键点
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        # 左眼中心点
        lef_center_x, lef_center_y = 0, 0
        for point in face_landmarks["left_eye"]:
            lef_center_x += point[0] * 4
            lef_center_y += point[1] * 4
        lef_center_x = int(lef_center_x / len(face_landmarks["left_eye"]))
        lef_center_y = int(lef_center_y / len(face_landmarks["left_eye"]))

        # 右眼中心点
        rig_center_x, rig_center_y = 0, 0
        for point in face_landmarks["right_eye"]:
            rig_center_x += point[0] * 4
            rig_center_y += point[1] * 4
        rig_center_x = int(rig_center_x / len(face_landmarks["right_eye"]))
        rig_center_y = int(rig_center_y / len(face_landmarks["right_eye"]))

        # 绘制眼睛和连接线
        cv2.circle(frame, (lef_center_x, lef_center_y), 25, (0, 0, 0), -1)
        cv2.circle(frame, (rig_center_x, rig_center_y), 25, (0, 0, 0), -1)
        cv2.line(frame, (rig_center_x, rig_center_y), (lef_center_x, lef_center_y), (0, 0, 0), 4)

        # 绘制眉毛区域
        cv2.ellipse(
            frame,
            (int((rig_center_x + lef_center_x) / 2), int((rig_center_y + lef_center_y) / 2 - 60)),
            (100, 25),
            0,
            0,
            360,
            (0, 0, 0),
            -1,
        )
        cv2.ellipse(
            frame,
            (int((rig_center_x + lef_center_x) / 2), int((rig_center_y + lef_center_y) / 2 - 80)),
            (60, 20),
            0,
            0,
            360,
            (0, 0, 0),
            -1,
        )
    
    # 保存结果为 JPG 文件
    output_path = "Test\lfw-mojing\\"+names[0]+"\\"+names[1]  # 设置保存路径和文件名
        # 自动创建文件夹
    output_dir = os.path.dirname(output_path)  # 获取文件夹路径
    if not os.path.exists(output_dir):  # 检查文件夹是否存在
        os.makedirs(output_dir)  # 如果不存在，则递归创建
    # 保存结果为 JPG 文件
    cv2.imwrite(output_path, frame)
    
