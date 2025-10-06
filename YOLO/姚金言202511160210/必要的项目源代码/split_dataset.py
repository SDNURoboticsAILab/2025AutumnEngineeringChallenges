import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_img_dir, source_lbl_dir, output_dir, split_ratio=0.8):
    """
    将源数据集随机划分为训练集和验证集，并移动到新的目录结构中。

    参数:
    - source_img_dir (Path): 存放所有源图片的文件夹路径。
    - source_lbl_dir (Path): 存放所有源标签的文件夹路径。
    - output_dir (Path): 用于存放划分后数据集的新文件夹的根路径。
    - split_ratio (float): 训练集所占的比例。
    """
    
    # 1. 创建输出目录结构
    train_images_path = output_dir / "images" / "train"
    val_images_path = output_dir / "images" / "val"
    train_labels_path = output_dir / "labels" / "train"
    val_labels_path = output_dir / "labels" / "val"
    
    print("正在创建输出目录...")
    for path in [train_images_path, val_images_path, train_labels_path, val_labels_path]:
        path.mkdir(parents=True, exist_ok=True)
    print("目录创建完成！")

    # 2. 获取所有图片文件列表
    image_files = sorted([p for p in source_img_dir.glob("*.jpg")])
    
    if not image_files:
        print(f"警告: 在目录 '{source_img_dir}' 中未找到任何 .jpg 文件。脚本将退出。")
        return

    # 3. 随机打乱文件列表
    random.seed(42)  # 设置随机种子以保证每次划分结果一致
    random.shuffle(image_files)
    
    # 4. 计算划分点
    split_point = int(len(image_files) * split_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    print(f"总共找到 {len(image_files)} 张图片。")
    print(f"划分为: {len(train_files)} 张训练图片, {len(val_files)} 张验证图片。")
    
    # 5. 移动文件
    print("\n正在移动训练集文件...")
    for image_path in tqdm(train_files, desc="移动训练集"):
        label_path = source_lbl_dir / (image_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(image_path), str(train_images_path))
            shutil.move(str(label_path), str(train_labels_path))
        else:
            print(f"警告: 找不到图片 '{image_path.name}' 对应的标签文件，已跳过。")
            
    print("\n正在移动验证集文件...")
    for image_path in tqdm(val_files, desc="移动验证集"):
        label_path = source_lbl_dir / (image_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(image_path), str(val_images_path))
            shutil.move(str(label_path), str(val_labels_path))
        else:
            print(f"警告: 找不到图片 '{image_path.name}' 对应的标签文件，已跳过。")

    print("\n数据集划分完成！")
    print(f"新的数据集已保存在: {output_dir}")

if __name__ == '__main__':
    # --- 配置区域 ---
    # 请根据您的实际情况修改以下路径
    
    # 1. 包含所有原始图片的文件夹 (ccpd_parser.py脚本的SOURCE_IMAGES_DIR)
    SOURCE_IMAGES_DIR = Path(r"D:\yolo\yolo\train\images")
    
    # 2. 包含所有原始标签的文件夹 (ccpd_parser.py脚本的OUTPUT_LABELS_DIR)
    SOURCE_LABELS_DIR = Path(r"D:\yolo\yolo\train\labels")
    
    # 3. 您希望创建并存放最终划分好的数据集的新文件夹
    #    建议使用 'dataset' 或 'License_Plate_Dataset' 等名称
    OUTPUT_DATASET_DIR = Path(r"D:\yolo\yolo\ultralytics\datasets\mydata")
    
    # 4. 训练集所占的比例
    TRAIN_RATIO = 0.8
    # --- 配置结束 ---
    
    # 运行主函数
    split_dataset(SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR, OUTPUT_DATASET_DIR, TRAIN_RATIO)