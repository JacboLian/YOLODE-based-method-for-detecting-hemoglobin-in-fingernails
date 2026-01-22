import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

def detect_and_classify(model, img_path, thresholds):
    """检测并根据阈值分类单张图片"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return {}

    results = model(img)
    pred = results[0].boxes.data.cpu().numpy() if hasattr(results[0], 'boxes') and results[0].boxes is not None else np.array([])
    pred_class_ids = pred[:, 5].astype(int) if pred.size > 0 else np.array([])
    total_boxes = len(pred_class_ids)
    normal_count = (pred_class_ids == 2).sum() if pred.size > 0 else 0

    image_results = {}
    for threshold in thresholds:
        classification = "unfit"
        if total_boxes > 0 and normal_count / total_boxes >= threshold:
            classification = "fit"
        image_results[threshold] = classification

    # image_results = {
    #     0.5:fit",
    #     0.6:"fit",
    #     0.7: "fit",
    #     0.8: "unfit",
    # }

    return image_results

def evaluate_dataset(model_path, image_dir, thresholds):
    """评估数据集，计算真阳率和真阴率"""
    model = YOLO(model_path)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    r_fit_count = 0
    r_unfit_count = 0
    results_data = []

    for file in image_files:  # 不需要索引 i
        img_path = os.path.join(image_dir, file)

        # 根据文件名判断真实标签
        if file.startswith("ab"):
            true_label = "r_unfit"
            r_unfit_count += 1
        elif file.startswith("no"):
            true_label = "r_fit"
            r_fit_count += 1
        else:
            print(f"警告: 文件名 {file} 不符合命名规则，跳过。")
            continue  # 跳过不符合规则的文件

        image_results = detect_and_classify(model, img_path, thresholds)
        for threshold, predicted_label in image_results.items():
            results_data.append([file, true_label, threshold, predicted_label])
        # [
        #     ["image0.jpg", "r_fit", 0.5, "fit"],
        #     ["image0.jpg", "r_fit", 0.6, "fit"],
        #     ["image0.jpg", "r_fit", 0.7, "fit"],
        #     ["image0.jpg", "r_fit", 0.8, "unfit"],
        # ]

    df = pd.DataFrame(results_data, columns=['Filename', 'True Label', 'Threshold', 'Predicted Label'])

    results = {}
    for threshold in thresholds:
        df_threshold = df[df['Threshold'] == threshold]
        true_negative = len(df_threshold[(df_threshold['True Label'] == 'r_fit') & (df_threshold['Predicted Label'] == 'fit')])
        true_positive = len(df_threshold[(df_threshold['True Label'] == 'r_unfit') & (df_threshold['Predicted Label'] == 'unfit')])

        tnr = true_negative / r_fit_count if r_fit_count > 0 else 0
        tpr = true_positive / r_unfit_count if r_unfit_count > 0 else 0
        results[threshold] = {"TPR": tpr, "TNR": tnr}

    return results

if __name__ == "__main__":
    model_path = './runs/train/custom_experiment_8s_0902_0.001_20018/weights/best.pt'
    image_dir = './TestDatasets'
    thresholds = [0.57, 0.7, 0.8]

    try:
        results = evaluate_dataset(model_path, image_dir, thresholds)
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.index.name = "Threshold"
        df_results.columns = ['真阳率(TPR)', '真阴率(TNR)']
        print(df_results)
        df_results.to_csv("./results.csv", encoding="utf-8-sig")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
