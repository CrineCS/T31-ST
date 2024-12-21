# 软件测试T31-代码作业

选题：面向人脸识别场景的深度学习模型测试技术

#### 小组成员

| 学号         | 姓名   |
| ------------ | ------ |
| 221250026    | 陈彦昀 |
| 522024320052 | 韩亚   |
| 522024320240 | 朱瑜婷 |
| 522024320172 | 谢淑娴 |

## 引言

##### 项目要求：

**在智能模型的测试中，不同场景的领域特性会对测试方法的设计带来而外的要求，例如自动驾驶场景下无意义的噪声添加 并不能代表现实生活中会遇到的天气、光照等真实问题。 该选题旨在让同学们掌握分析特定场景的能力，建立某一个特定场景下测试需求，设计并实现可用的测试方法。**

本次作业选取的是以人脸识别为模型的深度学习模型，启发于最近启用南园北门人脸识别门禁系统，希望对人脸识别模型开展测试，探究人脸识别系统在不同环境下是否能够正常运行。

使用的人脸识别框架：deepFace

(https://pypi.org/project/deepface/)



#### 依赖项

如缺少其他依赖请自行安装

- deepface 
- matplotlib
- PIL 
- cv2
- NumPy

#### 数据集

使用 Labeled Faces in the Wild (LFW) 数据集做基础测试/处理后测试评估:

- 13233 人脸图片
- 5749 人物身份
- 1680 人有两张以上照片

[LFW database](https://gitee.com/link?target=http%3A%2F%2Fvis-www.cs.umass.edu%2Flfw%2Flfw-funneled.tgz) 已经下载并放在 original_data 目录下

#### 如何运行

`python eval.py`

####  测试结果

在未做处理的情况下直接测得的准确度: **91 %**

## 场景测试

#### 光照处理

光照对于人脸识别的模型测试结果有较大影响

本部分代码主要通过模拟不同**方向**及**强度**的光照对数据集中图片进行处理，以模拟现实生活中会遇到的实际场景

具体实现 在 [light_simulator.py](https://github.com/CrineCS/T31-ST/blob/main/light_simulator.py) 文件当中, 通过OpenCV提供的函数对原图片打上不同强度不同方向的模拟光源

##### 参数

- brightness 表示图片受到光照影响后的亮度变化
  - 在实验中取值从0.1到5.0不均等分布
- light_angle 表示图片受到的光照的入射角度
  - 在实验中取0 45 90 135 180 225 270 315八个角度，即从图片的八个方位照射
- light_radius 表示图片受到的光照的半径大小
  - 在实验中固定为 200px，数据集中的图片大小为 250px * 250px

##### 效果

![image-20241221232251716](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232251716.png)![image-20241221232311672](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232311672.png)

​                                          处理前                                                                            处理后

##### 参数范围

brightness 在实验中取值

[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

angle 在实验中取值

[0, 45, 90, 135, 180, 225, 270, 315]

##### 结果

保存在[record.txt](https://github.com/CrineCS/T31-ST/blob/main/record.txt)中, 绘制的折线图如下：

![image-20241221231031628](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221231031628.png)

- 从图表中可以看出，在同等光照强度条件下，不同角度的光照对于准确率影响较小在同等光照角度时，亮度过暗或者过亮都会导致准确率降低
- 特别地，同等光照下225度入射角对结果影响最大而135度最小，推测原因为225度光对脸部的覆盖最大，135度光区位于左下角，覆盖脸部最小

#### 遮盖处理

遮盖对于人脸识别的模型测试结果有较大影响

本部分代码主要通过模拟墨镜和帽子的遮挡对数据集中图片进行处理，以模拟现实生活中会遇到的实际场景

具体实现 在 [add_mojing.py](https://github.com/CrineCS/T31-ST/blob/main/add_mojing.py) 文件当中, 通过在图片上添加黑色块进行模拟

##### 部分代码：

```python
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
        cv2.ellipse(frame,
            (int((rig_center_x + lef_center_x) / 2), int((rig_center_y + lef_center_y) / 2 - 60)),
            (100, 25),0,0,360,(0, 0, 0),-1,)
        cv2.ellipse(frame,
            (int((rig_center_x + lef_center_x) / 2), int((rig_center_y + lef_center_y) / 2 - 80)),
            (60, 20),0,0,360,(0, 0, 0),-1, )
```

##### 效果

![image-20241221232347761](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232347761.png)![image-20241221232352725](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232352725.png)

##### 结果

墨镜与帽子遮盖下模型的准确率约为59%，可以得出对于面部特征的直接遮挡（墨镜和帽子）会极大程度干扰模型的正常运行



#### 噪声处理

通过添加噪点，模拟现实中摄像头的问题，如传感器的噪声，镜头水滴、灰尘或摄像头焦距问题导致的模糊效果镜头表面的划痕或污渍

具体实现在[nosie_simulator,py](https://github.com/CrineCS/T31-ST/blob/main/nosie_simulator.py)中。

- 斑点污渍
- 划痕
- 指纹痕迹

##### 部分代码：

```python
def add_blurry_smudges(image, num_smudges=10, max_size=50, blur_radius=15):
    """
    添加模糊污点
    :param image: 处理图像
    :param num_smudges: 模糊污点数量
    :param max_size: 污点最大尺寸
    :param blur_radius: 模糊半径
    :return: 带污点的图像
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    for _ in range(num_smudges):
        shape_type = random.choice(['ellipse', 'rectangle'])
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        size = random.randint(10, max_size)
        if shape_type == 'ellipse':
            bbox = [x - size, y - size, x + size, y + size]
            color = (255, 255, 255, random.randint(50, 150))  # 半透明白色
            draw.ellipse(bbox, fill=color)
        else:
            bbox = [x - size, y - size, x + size, y + size]
            color = (255, 255, 255, random.randint(50, 150))  # 半透明白色
            draw.rectangle(bbox, fill=color)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
    blurred_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    image[mask == 255] = blurred_img[mask == 255]
    return image
```

##### 效果

![image-20241221232411244](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232411244.png)![image-20241221232416332](C:\Users\lx\AppData\Roaming\Typora\typora-user-images\image-20241221232416332.png)

##### 结果

噪声处理后模型准确率约为79%

对于面部特征的微小改动（噪声）并不能对模型的准确度产生太大的影响