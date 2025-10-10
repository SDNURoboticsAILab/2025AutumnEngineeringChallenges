一.安装Anaconda

下载：官网选择“Windows 64位 + Python 3.10”版本（避免版本冲突）

二.创建并激活“yolo_plate”虚拟环境

1. 创建虚拟环境
conda create -n yolo_plate python=3.10 -y
2. 激活环境
conda activate yolo_plate

   三.安装 PyTorch(有 NVIDIA)
      先查 CUDA 版本
      安装对应 PyTorch
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
      验证 GPU 配置成功
import torch
print(torch.cuda.is_available())  

四.安装 YOLO 依赖
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple