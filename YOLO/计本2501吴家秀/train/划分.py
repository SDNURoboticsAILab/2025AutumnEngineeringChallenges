import os
import shutil
import random

#1.定义路径
img_dir = "D:/yolo/train/images"  # 原始图像目录
txt_dir = "D:/yolo/train/labels"            # 原始标签目录
new_img_dir = "D:/yolo1/train"  # 划分后图像保存目录
new_txt_dir = "D:/yolo1/labels"  # 划分后标签保存目录

#2.创建新文件
os.makedirs(os.path.join(new_img_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(new_img_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(new_img_dir, "test"), exist_ok=True)
os.makedirs(os.path.join(new_txt_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(new_txt_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(new_txt_dir, "test"), exist_ok=True)

#3.图像与标签对应
txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
random.shuffle(txt_files)

#4.计算
total = len(txt_files)
train_num = int(total * 0.7)
val_num = int(total * 0.1)
test_num = total - train_num - val_num

# 5. 划分并复制文件
for i, txt_file in enumerate(txt_files):
    img_file = txt_file.replace(".txt", ".jpg")
    # 训练集
    if i < train_num:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(new_img_dir, "train", img_file))
        shutil.copy(os.path.join(txt_dir, txt_file), os.path.join(new_txt_dir, "train", txt_file))
    # 验证集
    elif i < train_num + val_num:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(new_img_dir, "val", img_file))
        shutil.copy(os.path.join(txt_dir, txt_file), os.path.join(new_txt_dir, "val", txt_file))
    # 测试集
    else:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(new_img_dir, "test", img_file))
        shutil.copy(os.path.join(txt_dir, txt_file), os.path.join(new_txt_dir, "test", txt_file))

print(f"划分完成！训练集{train_num}张，验证集{val_num}张，测试集{test_num}张")