import os
import xml.etree.ElementTree as ET
import shutil

very_low = []
low = []
normal = []
high = []
very_low_set = []
low_set = []
normal_set = []
high_set = []


def convert_xml_to_yolo(txt_output_dir, xml_input_dir):
    if not os.path.exists(txt_output_dir):
        os.makedirs(txt_output_dir, exist_ok=True)

    for file in os.listdir(xml_input_dir):
        if file.endswith('.xml'):
            xml_path = os.path.join(xml_input_dir, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_file = root.find('filename').text
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            txt_file = os.path.join(txt_output_dir, file.replace('.xml', '.txt'))
            with open(txt_file, 'w') as f:
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label == 'Very low':
                        class_id = 0
                        very_low.append(1)
                        very_low_set.append(txt_file)
                    elif label == 'low':
                        class_id = 1
                        low.append(1)
                        low_set.append(txt_file)
                    elif label == 'normal':
                        class_id = 2
                        normal.append(1)
                        normal_set.append(txt_file)
                    elif label == 'high':
                        class_id = 3
                        high.append(1)
                        high_set.append(txt_file)
                    else:
                        continue

                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

        elif file.endswith('.txt'):
            # 直接复制现有的YOLO格式标注文件
            src_txt_path = os.path.join(xml_input_dir, file)
            dst_txt_path = os.path.join(txt_output_dir, file)
            shutil.copy(src_txt_path, dst_txt_path)


if __name__ == "__main__":
    txt_output_dir = 'D:\zxb\301\yolov8\yolov8\datasets\data\labels'
    xml_input_dir = 'D:\zxb\301\301_nail_detection\301_nail_detection\mydatasets'
    convert_xml_to_yolo(txt_output_dir, xml_input_dir)
    print(len(very_low), len(low), len(normal), len(high))
