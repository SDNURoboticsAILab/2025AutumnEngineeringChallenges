from ultralytics import YOLO
import cv2 # 如果要进行实时摄像头推理，建议导入OpenCV
# 1. 加载您训练好的模型
# 请务必将这里的路径替换为您自己的 best.pt 文件的真实路径！
model = YOLO('runs/detect/train/weights/best.pt')

# 2. 指定要进行推理的图片
# 将 'path/to/your/test_image.jpg' 替换为你的测试图片路径
image_path = 'test_images/car_plate_01.jpg'

# 3. 执行推理
# save=True 会将带有检测框的图片保存下来
results = model.predict(source=image_path, save=True)

print("推理完成！请检查 runs/detect/predict 文件夹下的结果。")