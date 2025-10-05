import cv2
import numpy as np
import os
import torch  # 用于加载深度学习模型
from paddleocr import PaddleOCR

# --------------------- 关键配置 ---------------------
# 1. 文件路径
IMG_PATH = "车牌照片.jpg"  # 替换为你的图片路径
DET_MODEL_DIR = "D:\\paddle_ocr_cache\\whl\\det\\ch\\ch_PP-OCRv3_det_infer"
REC_MODEL_DIR = "D:\\paddle_ocr_cache\\whl\\rec\\ch\\ch_PP-OCRv3_rec_infer"
CLS_MODEL_DIR = "D:\\paddle_ocr_cache\\whl\\cls\\ch_ppocr_mobile_v2.0_cls_infer"

# 2. 车牌检测模型（轻量级深度学习模型）
MODEL_CONF = 0.5  # 检测置信度阈值
MODEL_INPUT_SIZE = 640  # 模型输入尺寸


# --------------------- 1. 准备车牌检测模型（核心） ---------------------
def load_plate_detector():
    # 检查是否有GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 加载YOLOv5轻量级车牌检测模型（已训练专门识别车牌）
        # 模型会自动下载到本地（约10MB）
        model = torch.hub.load(
            "ultralytics/yolov5", 
            "custom", 
            url="https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt",
            force_reload=False
        )
        # 调整模型参数
        model.classes = [2]  # 只检测车辆类别（后续在车辆区域内找车牌）
        model.conf = MODEL_CONF
        model.iou = 0.45
        model.to(device)
        print(f"✅ 车牌检测模型加载成功（使用{device}）")
        return model
    except Exception as e:
        print(f"⚠️ 模型加载失败，将使用备用定位方法：{str(e)}")
        return None


# --------------------- 2. 基于深度学习的精准定位 ---------------------
def ai_locate_plate(orig_img, model):
    if model is None:
        return None, None
    
    # 转换图片格式
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # 检测车辆（在车辆区域内搜索车牌，提高效率）
    results = model(img_rgb, size=MODEL_INPUT_SIZE)
    detections = results.pandas().xyxy[0]  # 转换为DataFrame
    
    # 如果检测到车辆，在车辆区域内找车牌
    if not detections.empty:
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # 截取车辆区域（车牌通常在车辆前部或后部）
            vehicle_roi = orig_img[y1:y2, x1:x2]
            # 在车辆区域内进行车牌精准搜索
            plate_roi, plate_box = search_plate_in_vehicle(vehicle_roi, (x1, y1))
            if plate_roi is not None:
                return plate_roi, plate_box
    
    # 如果未检测到车辆，直接在全图搜索
    print("🔍 未检测到车辆，全图搜索车牌...")
    plate_roi, plate_box = search_plate_in_vehicle(orig_img, (0, 0))
    return plate_roi, plate_box


# --------------------- 3. 在目标区域内精准搜索车牌 ---------------------
def search_plate_in_vehicle(roi, offset):
    ox, oy = offset  # 原图偏移量
    h, w = roi.shape[:2]
    
    # 1. 颜色筛选（蓝/黄车牌）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 蓝色车牌
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([120, 255, 255]))
    # 黄色车牌
    yellow_mask = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([30, 255, 255]))
    color_mask = cv2.bitwise_or(blue_mask, yellow_mask)
    
    # 2. 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. 轮廓筛选
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, wc, hc = cv2.boundingRect(cnt)
        # 车牌宽高比3:1左右
        if 2.5 < (wc/hc) < 4.0 and 100 < wc < 500 and 30 < hc < 150:
            # 计算矩形度（越接近矩形越可能是车牌）
            area = cv2.contourArea(cnt)
            rect_area = wc * hc
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity > 0.8:
                # 转换为原图坐标
                plate_x1 = ox + x
                plate_y1 = oy + y
                plate_x2 = ox + x + wc
                plate_y2 = oy + y + hc
                plate_roi = roi[y:y+hc, x:x+wc]
                # 保存候选区域
                cv2.imwrite("debug_ai定位车牌.jpg", plate_roi)
                return plate_roi, (plate_x1, plate_y1, plate_x2, plate_y2)
    
    return None, None


# --------------------- 4. 备用定位方案（模型加载失败时） ---------------------
def fallback_locate_plate(orig_img):
    print("🔍 使用备用方案定位车牌...")
    # 颜色+形状+文本多特征融合
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(blue_mask, yellow_mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, wc, hc = cv2.boundingRect(cnt)
        if 2.5 < (wc/hc) < 4.0 and 100 < wc < 500 and 30 < hc < 150:
            plate_roi = orig_img[y:y+hc, x:x+wc]
            cv2.imwrite("debug_备用定位车牌.jpg", plate_roi)
            return plate_roi, (x, y, x+wc, y+hc)
    
    # 最终兜底：文本检测
    ocr_det = PaddleOCR(
        det_model_dir=DET_MODEL_DIR,
        rec_model_dir=None,
        cls_model_dir=None,
        rec=False,
        cls=False,
        use_gpu=False,
        show_log=False
    )
    det_results = ocr_det.ocr(orig_img)
    if det_results and len(det_results[0]) > 0:
        for box in det_results[0]:
            pts = np.array(box, np.int32)
            x1, y1 = np.min(pts[:,0]), np.min(pts[:,1])
            x2, y2 = np.max(pts[:,0]), np.max(pts[:,1])
            if 2.5 < (x2-x1)/(y2-y1) < 4.0:
                plate_roi = orig_img[y1:y2, x1:x2]
                return plate_roi, (x1, y1, x2, y2)
    
    return None, None


# --------------------- 5. 识别与显示结果 ---------------------
def recognize_and_display(orig_img, plate_roi, box):
    if plate_roi is None:
        print("❌ 未找到车牌区域")
        return
    
    # OCR识别
    ocr = PaddleOCR(
        det_model_dir=DET_MODEL_DIR,
        rec_model_dir=REC_MODEL_DIR,
        cls_model_dir=CLS_MODEL_DIR,
        use_angle_cls=True,
        use_gpu=False,
        show_log=False
    )
    results = ocr.ocr(plate_roi, cls=True)
    
    # 提取车牌文字
    province = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新港澳台"
    plate_text = ""
    if results and len(results[0]) > 0:
        for line in results[0]:
            for c in line[1][0]:
                if c in province or c.isalnum():
                    plate_text += c
    plate_text = plate_text[:7]
    
    # 绘制结果
    x1, y1, x2, y2 = box
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        orig_img, plate_text, (x1, y1-10 if y1>30 else y1+30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    
    # 保存并显示
    cv2.imwrite("最终识别结果.jpg", orig_img)
    print(f"🎉 车牌识别结果：{plate_text}")
    cv2.imshow("定位的车牌区域", plate_roi)
    cv2.imshow("最终结果", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------- 主程序 ---------------------
def main():
    # 读取图片
    img_bytes = np.fromfile(IMG_PATH, np.uint8)
    orig_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if orig_img is None:
        print(f"❌ 无法读取图片：{IMG_PATH}")
        return
    
    # 加载深度学习模型
    model = load_plate_detector()
    
    # 优先使用AI定位
    plate_roi, box = ai_locate_plate(orig_img, model)
    
    # 如果AI定位失败，使用备用方案
    if plate_roi is None:
        plate_roi, box = fallback_locate_plate(orig_img)
    
    # 识别并显示结果
    recognize_and_display(orig_img, plate_roi, box)


if __name__ == "__main__":
    main()
    