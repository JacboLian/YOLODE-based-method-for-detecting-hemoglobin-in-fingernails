# Nail Detection & Classification Project

本项目基于 YOLOv8 实现目标的检测与分类（针对指甲或类似微小目标的 "Very low", "low", "normal", "high" 等级分类）。
项目包含了从数据清洗、格式转换、数据集划分、模型训练到模型评估的全套流程代码。

## 克隆仓库

```bash
git clone https://github.com/JacboLian/YOLODE-based-method-for-detecting-hemoglobin-in-fingernails.git
```

## 环境安装

请确保已安装 Anaconda 或 Miniconda。

```bash
conda env create -f environment.yaml
conda activate nail_detection_env
```

## 代码功能介绍
1. classification_dataset.py 
功能：数据清洗与分类数据集制作
检查图片： 自动扫描并删除损坏的 JPG/JPEG 图片。
解析 XML： 读取 VOC 格式的 XML 标注文件。
裁剪目标： 根据标注框（BBox）从原图中裁剪出目标物体，并增加一定的边缘填充（padding）。
按类别保存： 将裁剪后的图片按类别（Very low, low, normal, high）存入 classification_dataset/train 和 classification_dataset/val 文件夹中，用于训练分类模型或分析数据分布。

2. convert_to_yolo_format.py
功能：标签格式转换 (XML -> YOLO)
将 VOC 格式的 XML 标注文件转换为 YOLOv8 所需的 TXT 格式。
类别映射：
0: Very low
1: low
2: normal
3: high
归一化： 将坐标转换为 YOLO 要求的中心点坐标及宽高 (x_center, y_center, w, h)，且归一化到 0-1 之间。

3. split_dataset.py 
功能：划分训练集与验证集
将转换好的图片和 TXT 标签文件，按照指定比例（默认 9:1）随机划分为训练集（train）和验证集（val）。
文件会被移动到 yolov8/datasets/data/images 和 yolov8/datasets/data/labels 目录下，符合 YOLOv8 的标准目录结构。

4. train.py 
功能：模型训练
加载 YOLOv8 模型（如 yolov8n.pt）。
自动检测使用 GPU 或 CPU。
设置训练参数（Epochs, Batch Size, Learning Rate）。
开始训练并将结果保存到 runs/train 目录下。

5. get_best_datasets.py
功能：基于检测结果的数据筛选
解析日志： 读取 YOLO 的预测输出文件（如 pred_out.txt），利用正则表达式提取每张图的检测数量。
筛选： 找出检测框数量大于指定阈值（默认 5 个）的图片。
导出： 将符合要求的高质量数据（图片和对应标签）复制到新的文件夹（如 batter_datasets），用于下一轮的优化训练。

6. end_for_testdatasets.py
功能：模型指标评估
加载模型： 加载训练好的权重文件（.pt）。
业务逻辑评估： 对测试集图片进行推理。根据检测到的 normal 类别数量占比来判定整张图片是 "fit" (合格) 还是 "unfit" (不合格)。
计算指标： 在不同阈值（如 0.57, 0.7, 0.8）下，计算模型的 真阳率 (TPR) 和 真阴率 (TNR)。
结果输出： 将结果打印并保存为 results.csv。

## 运行流程
1. 数据清洗与裁剪
```bash
python classification_dataset.py 
```
2. 标签格式转换
```bash
python convert_to_yolo_format.py
```
3. 划分训练集与验证集
```bash
python split_dataset.py 
```
4. 模型训练
```bash
python train.py
```
5. 基于检测结果的数据筛选
```bash     
python get_best_datasets.py
```
6. 模型指标评估
```bash
python end_for_testdatasets.py
```
## 注意事项
本项目由于某些情况无法提供数据集，用户需自行准备数据集并按照代码要求的格式进行组织。
如有任何问题，欢迎提交 Issue 或联系作者 jacbolan44@gmail.com 。

## 贡献
欢迎提交 PR 或 Issue！

## 许可证
本项目遵循 MIT License。


