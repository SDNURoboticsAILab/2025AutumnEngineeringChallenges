
# -*- coding: utf-8 -*-

import warnings
import os
from ultralytics import YOLO
import cv2
import numpy as np

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont # 导入Pillow库

# 忽略一些不必要的警告信息
warnings.filterwarnings('ignore')


def process_plate_with_user_logic(image_data, ocr_engine, lang='ch'):

    full_text = ""
    try:
        if image_data is None or image_data.size == 0:
            print("错误：传入的图像数据为空")
            return "Error"
        print(f"    -> 传入OCR的图片尺寸：{image_data.shape[1]}x{image_data.shape[0]}")
        print("    -> 正在执行OCR识别...")
        result = ocr_engine.predict(input=image_data)
        if not result or not result[0]:
            print("    -> 未检测到任何文字")
            return "No Text"

        ocr_result = result[0]
        if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
            rec_texts = ocr_result['rec_texts']
        elif hasattr(ocr_result, 'rec_texts'):
            rec_texts = ocr_result.rec_texts
        else:
            return "Format Error"
        if not rec_texts:
            return "No Text"
        full_text = ''.join(rec_texts)
        print(f"    车牌号: {full_text}")
    except Exception as e:
        print(f"    OCR识别过程中发生错误：{e}")
        full_text = "OCR Error"
    return full_text

def draw_text_with_chinese(image, text, position, font_path, font_size):

    # 将OpenCV图像 (BGR) 转换为Pillow图像 (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 创建一个画笔
    draw = ImageDraw.Draw(img_pil)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except IOError:
        print(f"警告：无法加载字体文件 {font_path}。将使用默认字体（不支持中文）。")
        font = ImageFont.load_default()

    # 绘制文本 (白色)
    draw.text(position, text, font=font, fill=(255, 255, 255))

    # 将Pillow图像转换回OpenCV图像 (BGR)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv2


def main():

    # --- 1. 初始化模型 ---
    print("正在初始化YOLO模型...")
    yolo_model = YOLO(r'C:\Users\27800\runs\detect\train3\weights\best.pt')
    print("正在初始化OCR引擎...")
    ocr_engine = PaddleOCR(lang='ch')
    print("所有模型初始化成功。")

    # --- 2. ***** 重要配置区 ***** ---
    source_path = r'D:\yolo\yolo\test\images'
    output_project_path = r'detect'
    output_run_name = 'my_final_results'


    font_path = 'simhei.ttf'  # 确保这个字体文件和脚本在同一个目录下
    font_size = 32

    class_names = yolo_model.names
    PLATE_CLASS_NAME = 'license_plate'


    output_dir = os.path.join(output_project_path, output_run_name)
    os.makedirs(output_dir, exist_ok=True)



    results_iterator = yolo_model.predict(
        source=source_path, save=False, conf=0.25, iou=0.45, device=0, stream=True
    )

    # --- 4. 遍历检测结果并进行处理 ---
    for results in results_iterator:
        print(f"\n--- 正在处理图片: {results.path} ---")

        image_to_draw = results.orig_img.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                confidence = float(box.conf[0])
                print(f"  [检测到] 类别: '{class_name}', 置信度: {confidence:.2f}")

                bbox = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                label_text = class_name

                if class_name == PLATE_CLASS_NAME:
                    cropped_plate_img = results.orig_img[y1:y2, x1:x2]
                    plate_number = process_plate_with_user_logic(cropped_plate_img, ocr_engine)
                    if plate_number and plate_number not in ["No Text", "Error", "Format Error", "OCR Error"]:
                        label_text = plate_number



                cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)


                text_position_x = x1
                text_position_y = y1 - font_size - 5 if y1 - font_size - 5 > 0 else y1 + 5

                label_w, label_h = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0] # 估算一下背景大小
                cv2.rectangle(image_to_draw, (x1, text_position_y - 5), (x1 + label_w, text_position_y + font_size + 5), (0, 255, 0), cv2.FILLED)


                image_to_draw = draw_text_with_chinese(
                    image_to_draw,
                    label_text,
                    (text_position_x, text_position_y),
                    font_path,
                    font_size
                )


        output_image_path = os.path.join(output_dir, os.path.basename(results.path))
        cv2.imwrite(output_image_path, image_to_draw)
        print(f"  结果图已保存至: {output_image_path}")




if __name__ == '__main__':
    main()

