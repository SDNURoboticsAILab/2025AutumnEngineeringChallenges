# 导入所需的库
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import numpy as np

# --- 1. 初始化模型 ---
print("正在初始化YOLO模型...")
# 【请修改】确保这里的 'best.pt' 路径是正确的
model = YOLO('best.pt')

print("正在初始化OCR引擎...")
# lang='ch' 表示使用中文模型，use_angle_cls=True 表示自动矫正文字方向
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
print("所有模型初始化成功。")

# --- 2. 指定输入和输出 ---
# 【请修改】将这里替换为您想要测试的单张图片的完整路径
image_path = r'D:\yolo\yolo\test\images\your_test_image.jpg'
# 【可选】设置保存结果图片的文件名
output_path = 'result.jpg'

# --- 3. 读取并检查图片 ---
original_image = cv2.imread(image_path)
if original_image is None:
    print(f"错误：无法读取图片，请检查路径是否正确: {image_path}")
    exit()

# --- 4. 执行YOLOv8推理 ---
print(f"\n正在对图片进行目标检测: {image_path}")
# 直接对单张图片进行预测
results = model.predict(source=original_image)

# --- 5. 处理检测结果 ---
# predict返回的是一个列表，我们处理列表中的第一个结果
result = results[0]
if result.boxes is None or len(result.boxes) == 0:
    print("没有检测到任何车牌。")
else:
    # 遍历检测到的每一个边界框
    for box in result.boxes:
        # 获取边界框坐标
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords

        # 【核心步骤 1】裁剪车牌区域
        cropped_plate = original_image[y1:y2, x1:x2]

        # 【核心步骤 2】调用OCR模型进行字符识别
        ocr_result = ocr.ocr(cropped_plate, cls=True)

        # 【核心步骤 3】解析并绘制结果
        if ocr_result and ocr_result[0]:
            plate_text = ""
            # 将所有识别到的文字片段拼接成一个字符串
            for res in ocr_result[0]:
                plate_text += res[1][0]

            print(f"  [成功] 检测到车牌，识别结果: {plate_text}")

            # 在原始图像上绘制边界框 (绿色)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 使用OpenCV自带的putText绘制文字（简单，但可能不支持中文显示）
            # 如果您发现显示的是乱码或'??', 可以换回之前带Pillow的复杂版本
            cv2.putText(original_image, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("  [失败] 检测到车牌，但OCR未能识别出任何字符。")

# --- 6. 显示并保存最终结果 ---
print(f"\n处理完成，结果已保存到 {output_path}")
# 保存结果图片
cv2.imwrite(output_path, original_image)

# 显示带有检测框和识别结果的图像
cv2.imshow('YOLOv8 + OCR Result', original_image)

# 等待用户按键后关闭所有窗口 (按任意键即可退出)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("程序已退出。")