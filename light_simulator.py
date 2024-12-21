import os
import cv2
import numpy as np
import math


# 定义模拟光照的函数
def simulate_lighting(image, light_angle, light_radius, brightness):
    image = cv2.convertScaleAbs(image, alpha=brightness)
    height, width, _ = image.shape
    light_layer = np.zeros((height, width), dtype=np.float32)

    light_x = int(width / 2 + light_radius * math.cos(math.radians(light_angle)))
    light_y = int(height / 2 + light_radius * math.sin(math.radians(light_angle)))

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - light_x) ** 2 + (y - light_y) ** 2)
    intensity = 1 - np.clip(dist_from_center / light_radius, 0, 1)
    light_layer = (intensity * 255).astype(np.uint8)

    result = cv2.add(image, cv2.cvtColor(light_layer, cv2.COLOR_GRAY2BGR))
    return result


# 读取图像路径并处理
def process_images(input_txt, output_txt, output_dir, light_angles, light_radius, brightness_values):
    # 读取并清理掉空行
    with open(input_txt, 'r') as f:
        input_paths = [line.strip() for line in f.readlines() if line.strip()]  # 去除空行

    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个亮度和光照角度组合
    for brightness in brightness_values:
        for angle in light_angles:
            output_paths = []
            output_dir_for_combination = f"{output_dir}_light_{brightness}_{angle}"
            os.makedirs(output_dir_for_combination, exist_ok=True)

            # 更新output.txt的文件路径
            output_txt_for_combination = f"{output_txt}_light_{brightness}_{angle}.txt"

            for input_path in input_paths:
                image_path = os.path.join('dataset', 'lfw_funneled', input_path)  # 拼接完整路径
                print(f"Reading image from: {image_path}")  # 输出拼接后的路径

                # 读取图像
                image = cv2.imread(image_path)

                if image is None:
                    print(f"图像 {image_path} 无法读取!")
                    continue

                # 模拟光照
                result_image = simulate_lighting(image, angle, light_radius, brightness)

                # 构建输出路径
                output_image_path = os.path.join(output_dir_for_combination, input_path)
                output_image_dir = os.path.dirname(output_image_path)
                os.makedirs(output_image_dir, exist_ok=True)

                cv2.imwrite(output_image_path, result_image)

                output_paths.append(input_path)

            # 保存输出路径到 output.txt 文件
            with open(output_txt_for_combination, 'w') as f:
                for output_path in output_paths:
                    f.write(output_path + '\n')

            print(f"处理完毕，输出路径已保存至 {output_txt_for_combination}")


# 输入输出路径
input_txt = 'dataset/lfw_funneled/pairs_01.txt'  # 输入文件路径
output_txt = 'output_paths_light'  # 输出文件路径的前缀
output_dir = 'processed_images_light'  # 处理后的图像保存文件夹的前缀

# 模拟光照参数
light_radius = 100  # 光源半径
brightness_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 3.0]  # brightness取值范围
light_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # angle取值范围

# 调用处理函数
process_images(input_txt, output_txt, output_dir, light_angles, light_radius, brightness_values)
