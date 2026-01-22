import os
import random
import shutil


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def split_dataset(dataset_dir, label_dir, train_ratio=0.9):
    images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png')) and os.path.exists(
        os.path.join(label_dir, f.replace('.jpg', '.txt').replace('.png', '.txt')))]
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    train_image_dir = 'yolov8/datasets/data/images/train'
    val_image_dir = 'yolov8/datasets/data/images/val'
    train_label_dir = 'yolov8/datasets/data/labels/train'
    val_label_dir = 'yolov8/datasets/data/labels/val'

    # 清空目标目录
    clear_directory(train_image_dir)
    clear_directory(val_image_dir)
    clear_directory(train_label_dir)
    clear_directory(val_label_dir)

    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(val_image_dir):
        os.makedirs(val_image_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)

    for img in train_images:
        shutil.copy(os.path.join(dataset_dir, img), os.path.join(train_image_dir, img))
        shutil.copy(os.path.join(label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt')),
                    os.path.join(train_label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt')))

    for img in val_images:
        shutil.copy(os.path.join(dataset_dir, img), os.path.join(val_image_dir, img))
        shutil.copy(os.path.join(label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt')),
                    os.path.join(val_label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt')))


if __name__ == "__main__":
    dataset_dir = 'mydatasets'
    label_dir = 'yolov8/datasets/data/labels'
    split_dataset(dataset_dir, label_dir)
