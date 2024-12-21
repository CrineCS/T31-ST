import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import shutil


# 模拟不同类型的噪声
def add_spotty_stains(image, num_spots=100, spot_size=2):
    """
    添加斑点污渍
    :param image: 处理图像
    :param num_spots: 斑点数量
    :param spot_size: 斑点大小
    :return: 带污渍的图像
    """
    noisy_image = image.copy()
    for _ in range(num_spots):
        x = random.randint(0, noisy_image.shape[1] - 1)
        y = random.randint(0, noisy_image.shape[0] - 1)
        color = random.choice([(255, 255, 255), (0, 0, 0)])  # 白色或黑色
        cv2.circle(noisy_image, (x, y), random.randint(1, spot_size), color, -1)
    return noisy_image


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


def add_scratches(image, num_scratches=5, max_length=100, thickness=2):
    """
    添加划痕
    :param image: 处理图像
    :param num_scratches: 划痕数量
    :param max_length: 划痕最大长度
    :param thickness: 划痕厚度
    :return: 带划痕的图像
    """
    noisy_image = image.copy()
    for _ in range(num_scratches):
        x1 = random.randint(0, noisy_image.shape[1] - 1)
        y1 = random.randint(0, noisy_image.shape[0] - 1)
        angle = random.uniform(0, 360)
        length = random.randint(20, max_length)
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = int(y1 + length * np.sin(np.radians(angle)))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(noisy_image, (x1, y1), (x2, y2), color, thickness)
    noisy_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    return noisy_image


def add_fingerprint_marks(image, num_marks=5, max_size=30, blur_radius=10):
    """
    添加指纹痕迹
    :param image: 处理图像
    :param num_marks: 指纹痕迹数量
    :param max_size: 指纹痕迹最大尺寸
    :param blur_radius: 模糊半径
    :return: 带指纹痕迹的图像
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    for _ in range(num_marks):
        points = []
        num_points = random.randint(5, 15)
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        size = random.randint(10, max_size)
        for _ in range(num_points):
            offset_x = random.randint(-size, size)
            offset_y = random.randint(-size, size)
            points.append((x + offset_x, y + offset_y))
        color = (255, 255, 255, random.randint(50, 150))  # 半透明白色
        draw.polygon(points, fill=color)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
    fingerprint_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(fingerprint_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    image[mask == 255] = fingerprint_img[mask == 255]
    return image


# 读取输入的路径文件，随机选择噪声类型并处理图像
def process_images(input_txt, output_txt, output_dir):
    # 确保处理后图像保存的目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取输入文件路径
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    # 去除空行并添加前缀
    paths = [line.strip() for line in lines if line.strip()]

    processed_paths = []

    for path in paths:
        # 读取图像路径
        image_path = f"dataset/lfw_funneled/{path}"

        # 读取图像
        image = cv2.imread(image_path)

        # 随机选择多个噪声类型
        noise_types = random.sample(['spotty_stains', 'blurry_smudges', 'scratches', 'fingerprint_marks'],
                                    random.randint(1, 4))  # 随机选择1至4种噪声

        # 根据选择的噪声类型依次调用对应的函数
        for noise_type in noise_types:
            if noise_type == 'spotty_stains':
                image = add_spotty_stains(image)
            elif noise_type == 'blurry_smudges':
                image = add_blurry_smudges(image)
            elif noise_type == 'scratches':
                image = add_scratches(image)
            elif noise_type == 'fingerprint_marks':
                image = add_fingerprint_marks(image)

        # 生成处理后图像的保存路径，不加前缀
        relative_path = os.path.dirname(path)  # 保持文件夹路径结构
        processed_image_path = os.path.join(output_dir, relative_path)

        # 确保输出路径的文件夹存在
        if not os.path.exists(processed_image_path):
            os.makedirs(processed_image_path)

        # 保存处理后的图像
        image_name = os.path.basename(path)
        cv2.imwrite(os.path.join(processed_image_path, image_name), image)

        # 记录保存路径，保持与输入相同的路径结构
        processed_paths.append(f"{relative_path}/{image_name}")

    # 保存路径到输出文件
    with open(output_txt, 'w') as f:
        for i, path in enumerate(processed_paths):
            f.write(f"{path}\n")
            # 每4个路径后插入空行
            if (i + 1) % 4 == 0:
                f.write("\n")


# 运行示例
input_txt = 'dataset/lfw_funneled/pairs_01.txt'  # 输入路径的文件
output_txt = 'output_paths.txt'  # 输出文件路径
output_dir = 'processed_images'  # 处理后图像保存的文件夹

process_images(input_txt, output_txt, output_dir)


