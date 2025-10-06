import os
import cv2
from tqdm import tqdm # 引入tqdm库用于显示进度条

def ccpd_file_parser(image_path):
    """
    解析CCPD数据集的单个文件名，提取边界框坐标。
    文件名格式示例: 025-95_113-154&383_386&473-386&473_154&383_95_113_363_323-0_0_2_27_28_33_31-73_168.jpg
    我们需要的是第三个'-'分割的部分: 154&383_386&473 (左下角和右上角坐标)
    """
    try:
        # 1. 按'-'分割文件名
        parts = image_path.stem.split('-')
        
        # 2. 获取边界框坐标部分
        bbox_part = parts[2]
        
        # 3. 按'_'分割，获取两个对角点坐标
        points = bbox_part.split('_')
        
        # 4. 解析第一个点 (左下角)
        x1_str, y1_str = points[0].split('&')
        x1, y1 = int(x1_str), int(y1_str)
        
        # 5. 解析第二个点 (右上角)
        x2_str, y2_str = points[1].split('&')
        x2, y2 = int(x2_str), int(y2_str)
        
        return [x1, y1, x2, y2]
        
    except (ValueError, IndexError) as e:
        print(f"文件名格式错误，无法解析: {image_path.name}. 错误: {e}")
        return None


def convert_to_yolo_format(bbox, img_width, img_height):
    """
    将[x1, y1, x2, y2]格式的坐标转换为YOLO格式。
    YOLO格式: <class_id> <x_center> <y_center> <width> <height>
    所有值都是相对于图像宽高的归一化值。
    """
    # CCPD是单一类别检测，所以class_id永远是0
    class_id = 0
    
    # 计算边界框的宽度和高度
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    
    # 计算边界框的中心点坐标
    center_x = bbox[0] + box_w / 2
    center_y = bbox[1] + box_h / 2
    
    # 归一化
    yolo_x = center_x / img_width
    yolo_y = center_y / img_height
    yolo_w = box_w / img_width
    yolo_h = box_h / img_height
    
    return f"{class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}"


def process_ccpd_dataset(source_dir, output_dir):
    """
    处理整个CCPD数据集文件夹。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件的路径
    image_files = list(Path(source_dir).glob("*.jpg"))
    
    if not image_files:
        print(f"警告: 在目录 '{source_dir}' 中未找到任何 .jpg 文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # 使用tqdm创建进度条
    for image_path in tqdm(image_files, desc="处理图片"):
        # 1. 读取图片获取尺寸
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"\n警告: 无法读取图片 {image_path.name}, 跳过。")
                continue
            img_height, img_width, _ = img.shape
        except Exception as e:
            print(f"\n读取图片 {image_path.name} 时出错: {e}, 跳过。")
            continue

        # 2. 解析文件名获取边界框
        bbox = ccpd_file_parser(image_path)
        
        if bbox:
            # 3. 转换为YOLO格式
            yolo_string = convert_to_yolo_format(bbox, img_width, img_height)
            
            # 4. 构造标签文件名并保存
            label_filename = output_dir / (image_path.stem + ".txt")
            with open(label_filename, 'w') as f:
                f.write(yolo_string)

    print("\n处理完成！")
    print(f"所有标签文件已保存在: {output_dir}")


if __name__ == '__main__':
    from pathlib import Path
    
    # --- 配置区域 ---
    # 1. 设置您的CCPD图片所在的源文件夹
    #    请确保使用正斜杠'/'或双反斜杠'\\'
    SOURCE_IMAGES_DIR = r"D:\yolo\yolo\train\images"
    
    # 2. 设置您希望保存.txt标签文件的目标文件夹
    OUTPUT_LABELS_DIR = r"D:\yolo\yolo\train\labels"
    # --- 配置结束 ---

    # 检查tqdm是否已安装
    try:
        from tqdm import tqdm
    except ImportError:
        print("警告: tqdm 库未安装。进度条将不可用。")
        print("您可以通过 'pip install tqdm' 来安装它。")
        # 创建一个虚拟的tqdm函数，以便脚本在没有tqdm的情况下也能运行
        def tqdm(iterable, *args, **kwargs):
            return iterable

    # 运行主处理函数
    process_ccpd_dataset(Path(SOURCE_IMAGES_DIR), Path(OUTPUT_LABELS_DIR))```