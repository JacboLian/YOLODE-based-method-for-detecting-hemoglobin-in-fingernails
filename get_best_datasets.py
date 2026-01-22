import os
import shutil
import re
import argparse
from pathlib import Path

import re

def find_images_with_detections(yolo_output_file, min_detections=5):
    """从YOLOv8输出文件中找出检测到目标框数量超过阈值的图片索引。"""
    indices = []
    try:
        with open(yolo_output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # 更健壮的正则表达式，处理只有一种检测框的情况
                match = re.match(r"^0: \d+x\d+ (?:(\d+) (lows?|normals?), )?(?:(\d+) (lows?|normals?), )?.*", line)
                if match:
                    counts = {}
                    for j in range(0, 4, 2):  # 遍历匹配到的所有数量和类型
                        count_str = match.group(j + 1)
                        type_str = match.group(j + 2)
                        if count_str and type_str:
                            counts[type_str] = int(count_str)
                    total_count = sum(counts.values())
                    if total_count > min_detections:
                        indices.append(i)
                elif "(no detections)" in line:
                    continue  # 跳过没有检测到的行
                else:
                    print(f"警告: 第 {i+1} 行格式不匹配: {line.strip()}")  # 输出不匹配的行，方便调试
    except FileNotFoundError:
        print(f"错误：找不到YOLO输出文件：{yolo_output_file}")
        return []
    except Exception as e:
        print(f"读取YOLO输出文件时发生错误：{e}")
        return []
    return indices

def copy_images(image_directory, output_image_directory, label_directory, output_label_directory, indices):
    """根据索引列表复制图片和标签到各自的输出目录。"""

    output_image_path = Path(output_image_directory)
    output_image_path.mkdir(parents=True, exist_ok=True)

    output_label_path = Path(output_label_directory) #创建标签输出目录
    output_label_path.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_directory)
    label_path = Path(label_directory)

    try:
        image_files = sorted([f.name for f in image_path.iterdir() if f.is_file()])
        image_index_map = dict(enumerate(image_files))

        label_files = sorted([f.name for f in label_path.iterdir() if f.is_file()])
        label_index_map = dict(enumerate(label_files))

        if len(image_files) != len(label_files):
            print("警告：图片文件数量和标签文件数量不一致，可能导致索引错误。")

    except FileNotFoundError:
        print(f"错误：找不到图片或标签目录。")
        return
    except Exception as e:
        print(f"读取图片或标签目录时发生错误：{e}")
        return

    for index in indices:
        if index in image_index_map and index in label_index_map:
            source_image = image_path / image_index_map[index]
            dest_image = output_image_path / image_index_map[index]

            source_label = label_path / label_index_map[index]
            dest_label = output_label_path / label_index_map[index] #使用标签输出路径

            try:
                shutil.copy2(source_image, dest_image)
                print(f"复制图片：{source_image} 到 {dest_image}")

                shutil.copy2(source_label, dest_label)
                print(f"复制标签：{source_label} 到 {dest_label}")

            except Exception as e:
                print(f"复制文件时发生错误：{e}")
        else:
            print(f"警告：索引 {index} 超出图片或标签范围。可用索引范围为 0 到 {min(len(image_index_map), len(label_index_map)) - 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从YOLOv8输出中提取检测框数量大于指定值的图片和标签。")
    parser.add_argument("yolo_output", nargs='?', default='./pred_out.txt', help="YOLOv8输出文件路径")
    parser.add_argument("image_dir", nargs='?', default='./datasets/data/train/train', help="原始图片目录")
    parser.add_argument("label_dir", nargs='?', default='./datasets/data/train/train', help="原始标签目录")
    parser.add_argument("-oi", "--output_image_dir", default="./batter_datasets/batter_data/train", help="输出图片目录") #添加图片输出目录参数
    parser.add_argument("-ol", "--output_label_dir", default="./batter_datasets/batter_data/train", help="输出标签目录") #添加标签输出目录参数
    parser.add_argument("-m", "--min_detections", type=int, default=5, help="最小检测框数量")
    args = parser.parse_args()

    indices = find_images_with_detections(args.yolo_output, args.min_detections)
    if indices:
        print(f"检测框数量大于 {args.min_detections} 的图片索引: {indices}")
        copy_images(args.image_dir, args.output_image_dir, args.label_dir, args.output_label_dir, indices) #传递标签输出目录参数
        print("complete!")
    else:
        print("can not find effective imgs or Yolo output file format error")