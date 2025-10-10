import os
import xml.etree.ElementTree as ET

# 定义类别列表（根据你的数据集调整）
classes = ["car_id"]  # 替换成你的实际类别

# XML文件夹路径
xml_dir = r'C:\Users\86135\Desktop\YOLO\xml'
# 输出TXT文件夹路径
txt_dir = r'C:\Users\86135\Desktop\YOLO\txt'

# 确保输出文件夹存在
os.makedirs(txt_dir, exist_ok=True)

# 遍历XML文件
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片尺寸（用于归一化）
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 创建对应的TXT文件
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue  # 跳过未定义的类别
                class_id = classes.index(cls)

                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # 计算YOLO格式的归一化坐标
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                # 写入TXT文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print("转换完成！")