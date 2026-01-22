import torch
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(data_config, model_config, epochs=100, batch_size=16, learning_rate=0.001):
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')
    else:
        print("CUDA is available. Using GPU.")
        device = torch.device('cuda')
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    print(f'Using device: {device}')


    # 加载YOLO模型
    model = YOLO(model_config).to(device)

    # 训练模型，并指定保存路径和early stopping参数
    # project：存储训练结果的路径
    # name：保存的实验名称，包含学习率和轮次信息

    # name = 'custom_experiment_8s_0902_' + str(learning_rate) + '_' + str(epochs)
    name='custom_experiment_lian' + str(learning_rate) + '_' + str(epochs)

    model.train(data=data_config, epochs=epochs, batch=batch_size, device=device, lr0=learning_rate,
                project='runs/train',
                name=name)


if __name__ == "__main__":
    data_config = './data/dataset.yaml'
    model_config = './yolov8n.pt'
    epochs = 200  # 设置训练的epoch数量
    learning_rate = 0.001
    train(data_config, model_config, epochs=epochs, learning_rate=learning_rate)
