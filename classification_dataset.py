import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import random
from tqdm import tqdm
from collections import defaultdict


def check_and_fix_images(image_dir):
    """
    检查并删除损坏的 JPEG 图片
    """
    print("Checking and fixing train...")
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # 验证图片完整性
                except (IOError, SyntaxError):
                    print(f"Corrupted image found and removed: {file_path}")
                    os.remove(file_path)  # 删除损坏的图片
    print("Image check and fix completed.")


def parse_xml(xml_file):
    """
    解析 XML 文件，获取边界框及其类别
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")

        try:
            bbox = {
                "xmin": int(float(bndbox.find("xmin").text.strip())),
                "ymin": int(float(bndbox.find("ymin").text.strip())),
                "xmax": int(float(bndbox.find("xmax").text.strip())),
                "ymax": int(float(bndbox.find("ymax").text.strip())),
            }
            print(f"Parsed bbox: {bbox} in file {xml_file}")  # 调试打印
            objects.append({"name": name, "bbox": bbox})
        except (ValueError, AttributeError) as e:
            print(f"Error parsing bounding box in file {xml_file}: {e}")
            continue

    return objects


def adjust_bbox(bbox, image_size, padding=0.1):
    """
    调整边界框，保留一定的背景区域，并验证数值合理性
    """
    try:
        # 强制将 bbox 值转换为整数
        xmin, ymin, xmax, ymax = (int(bbox[key]) for key in ["xmin", "ymin", "xmax", "ymax"])

        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid bounding box dimensions: {bbox}")

        pad_w = int(padding * width)
        pad_h = int(padding * height)

        xmin = max(0, xmin - pad_w)
        ymin = max(0, ymin - pad_h)
        xmax = min(image_size[0], xmax + pad_w)
        ymax = min(image_size[1], ymax + pad_h)

        return xmin, ymin, xmax, ymax
    except Exception as e:
        print(f"Error in adjust_bbox with bbox: {bbox}, image_size: {image_size}, error: {e}")
        raise


def clear_output_directory(output_dir):
    """
    清空输出目录
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除整个目录
    os.makedirs(output_dir)  # 重新创建空目录
    print(f"Cleared and recreated directory: {output_dir}")


def create_classification_dataset(mydatasets_dir, output_dir, split_ratio=0.9, padding=0.15):
    """
    创建分类数据集：裁剪图片并存储到分类目录中，并打印统计信息
    """
    # 定义类别
    categories = ["Very low", "low", "normal", "high"]
    for split in ["train", "val"]:
        for category in categories:
            path = os.path.join(output_dir, split, category)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

    # 获取所有图片和对应的标注
    all_data = defaultdict(list)
    for file in os.listdir(mydatasets_dir):
        if file.endswith(".xml"):
            base_name = os.path.splitext(file)[0]
            image_file = os.path.join(mydatasets_dir, base_name + ".jpg")
            xml_file = os.path.join(mydatasets_dir, file)
            if os.path.exists(image_file):
                objects = parse_xml(xml_file)
                for obj in objects:
                    all_data[obj["name"]].append({"image": image_file, "bbox": obj["bbox"]})

    # 打印总分类统计信息
    print("\nTotal train per category:")
    for category in categories:
        print(f"  {category}: {len(all_data[category])} train")

    # 随机打乱数据并按比例划分
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    for category in categories:
        data = all_data[category]
        random.shuffle(data)
        split_index = int(len(data) * split_ratio)
        train_data[category] = data[:split_index]
        val_data[category] = data[split_index:]

    # 裁剪并保存图片
    for split, data_dict in [("train", train_data), ("val", val_data)]:
        for category, items in data_dict.items():
            for item in tqdm(items, desc=f"Processing {split} data for {category}"):
                print(f"Processing item: {item}")  # 调试打印
                img = Image.open(item["image"])
                image_size = img.size

                # 打印 bbox 值和类型
                print(
                    f"Before adjust_bbox: {item['bbox']} ({type(item['bbox']['xmin'])}, {type(item['bbox']['ymin'])}, {type(item['bbox']['xmax'])}, {type(item['bbox']['ymax'])})")

                bbox = adjust_bbox(item["bbox"], image_size, padding=padding)
                cropped = img.crop(bbox)

                # 保存到对应类别文件夹
                save_dir = os.path.join(output_dir, split, category)
                save_path = os.path.join(save_dir, f"{os.path.basename(item['image'])}_{category}.jpg")
                cropped.save(save_path)

    # 打印训练集和验证集统计信息
    print("\nDataset split statistics:")
    for category in categories:
        print(f"  {category}:")
        print(f"    Train: {len(train_data[category])} train")
        print(f"    Val:   {len(val_data[category])} train")

    print("Classification dataset created successfully.")


if __name__ == "__main__":
    # 输入和输出路径
    mydatasets_dir = "../mydatasets"
    output_dir = "classification_dataset"

    # 清空输出目录
    clear_output_directory(output_dir)

    # 检查并修复损坏的图片
    check_and_fix_images(mydatasets_dir)

    # 创建分类数据集
    create_classification_dataset(mydatasets_dir, output_dir, split_ratio=0.9, padding=0.15)
