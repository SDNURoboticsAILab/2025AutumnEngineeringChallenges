import cv2
import os
import numpy as np
from ultralytics import YOLO  # 新增：导入YOLO模型库
from paddleocr import PaddleOCR
from PIL import Image

# 1. 加载模型（核心修改：优先用YOLO检测车牌）
# 替换为你的YOLO训练模型路径（best.pt），确保路径正确
YOLO_MODEL_PATH = "runs/train/exp/weights/best.pt"  # 常见路径1
# YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"  # 常见路径2，二选一
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ 成功加载YOLO车牌检测模型：{YOLO_MODEL_PATH}")
except Exception as e:
    print(f"❌ 加载YOLO模型失败！请检查路径是否正确：{e}")
    exit()

# 2. 初始化OCR（字符识别，保持不变）
ocr = PaddleOCR(
    lang='ch',
    det_model_dir='D:\\ocr_models\\det\\ch_PP-OCRv3_det_infer',
    rec_model_dir='D:\\ocr_models\\rec\\ch_PP-OCRv3_rec_infer',
    use_angle_cls=False
)

# 3. 配置路径和统计
image_folder = 'D:\\模型\\第五步\\images'  # 你的113张图片文件夹
output_folder = 'D:\\模型\\第五步\\yolo_results'  # 结果保存文件夹
os.makedirs(output_folder, exist_ok=True)  # 自动创建文件夹，避免报错

success_count = 0
fail_count = 0
result_log = []  # 记录每张图片的结果

# 4. 收集所有图片路径
image_paths = [
    os.path.join(image_folder, f) 
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
]

if not image_paths:
    print(f"❌ 在 {image_folder} 中未找到图片")
    exit()
print(f"✅ 找到 {len(image_paths)} 张图片，开始处理...\n")

# 5. 批量处理（YOLO检测 + OCR识别）
for idx, img_path in enumerate(image_paths, 1):
    img_name = os.path.basename(img_path)
    print(f"===== 处理进度：{idx}/{len(image_paths)} - {img_name} =====")
    
    # 读取图片（兼容特殊格式）
    img = None
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img_pil = Image.open(img_path).convert('RGB')
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            print("⚠️ 已用PIL兼容模式读取图片")
    except Exception as e:
        print(f"❌ 图片读取失败：{str(e)[:50]}\n")
        fail_count += 1
        result_log.append(f"{img_name}: 读取失败 - {str(e)[:30]}")
        continue

    # 6. 用YOLO检测车牌（核心步骤，替换之前的自定义定位）
    plate_box = None  # 存储车牌坐标 (x1, y1, x2, y2)
    try:
        # 检测车牌，置信度阈值0.3（过滤模糊结果）
        yolo_results = yolo_model(img, conf=0.3, verbose=False)  # verbose=False关闭多余输出
        # 提取第一个检测到的车牌（默认一张图一个车牌）
        if len(yolo_results[0].boxes) > 0:
            # 获取边界框坐标（x1, y1是左上角，x2, y2是右下角）
            box = yolo_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            # 适当扩展边界框，避免裁剪到字符边缘
            expand = 6
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(img.shape[1], x2 + expand)
            y2 = min(img.shape[0], y2 + expand)
            plate_box = (x1, y1, x2, y2)
            print(f"⚠️ 成功检测到车牌区域：({x1}, {y1}) 到 ({x2}, {y2})")
        else:
            print("❌ YOLO未检测到车牌区域\n")
            fail_count += 1
            result_log.append(f"{img_name}: 未检测到车牌")
            continue
    except Exception as e:
        print(f"❌ YOLO检测出错：{str(e)[:50]}\n")
        fail_count += 1
        result_log.append(f"{img_name}: 检测出错 - {str(e)[:30]}")
        continue

    # 7. 裁剪车牌区域并预处理
    x1, y1, x2, y2 = plate_box
    plate_img = img[y1:y2, x1:x2]
    # 预处理增强（提高OCR识别率）
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    plate_enhanced = clahe.apply(plate_gray)  # 增强对比度

    # 8. OCR识别车牌字符
    try:
        ocr_result = ocr.ocr(plate_enhanced, det=False, rec=True)
        plate_text = ""
        max_confidence = 0.0

        # 解析OCR结果（兼容不同版本格式）
        if ocr_result and len(ocr_result) > 0:
            for item in ocr_result:
                if isinstance(item, list) and len(item) > 0:
                    line = item[0]
                    if isinstance(line, tuple) and len(line) >= 2:
                        text = line[0] if isinstance(line[0], str) else ""
                        conf = line[1] if isinstance(line[1], (int, float)) else 0.0
                        conf = float(conf)
                        # 车牌通常6-7个字符，优先选择符合长度的结果
                        if (len(text) == 6 or len(text) == 7) and conf > max_confidence:
                            max_confidence = conf
                            plate_text = text

        # 验证识别结果
        if plate_text and max_confidence > 0.45:
            print(f"✅ 识别成功！车牌：{plate_text}（置信度：{max_confidence:.2f}）\n")
            success_count += 1
            result_log.append(f"{img_name}: 成功 - {plate_text}（{max_confidence:.2f}）")
            
            # 在原图标注车牌框和结果，保存图片
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            cv2.putText(img, plate_text, (x1, y1-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, f"result_{img_name}"), img)
        else:
            print(f"❌ 识别失败（字符：{plate_text}，置信度：{max_confidence:.2f}）\n")
            fail_count += 1
            result_log.append(f"{img_name}: 识别失败 - {max_confidence:.2f}")
            # 保存增强后的车牌图，方便分析原因
            cv2.imwrite(os.path.join(output_folder, f"failed_plate_{img_name}"), plate_enhanced)
    except Exception as e:
        print(f"❌ OCR识别出错：{str(e)[:50]}\n")
        fail_count += 1
        result_log.append(f"{img_name}: 识别出错 - {str(e)[:30]}")
        continue

# 9. 生成结果统计日志
with open(os.path.join(output_folder, "处理结果汇总.txt"), "w", encoding="utf-8") as f:
    f.write(f"车牌识别批量处理结果\n")
    f.write(f"总图片数：{len(image_paths)} 张\n")
    f.write(f"成功识别：{success_count} 张\n")
    f.write(f"处理失败：{fail_count} 张\n")
    f.write(f"成功率：{success_count/len(image_paths)*100:.1f}%\n\n")
    f.write("详细结果列表：\n")
    for i, log in enumerate(result_log, 1):
        f.write(f"{i}. {log}\n")

# 10. 打印最终统计
print("\n" + "="*60)
print(f"🎉 所有图片处理完成！")
print(f"📁 结果文件保存在：{output_folder}")
print(f"✅ 成功识别：{success_count} 张 | ❌ 处理失败：{fail_count} 张")
print(f"📊 整体成功率：{success_count/len(image_paths)*100:.1f}%")
print("="*60)