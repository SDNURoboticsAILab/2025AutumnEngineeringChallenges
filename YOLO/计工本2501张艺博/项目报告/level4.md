一.准备测试数据
图片测试集：在 D:/YOLO_plate_dataset 新建 test_imgs 文件夹，放入 非训练集的车牌图片（从网上下载的46张）。
视频测试集：准备 1 段含车牌的视频（test_video.mp4，从文件夹中找到的），放在 D:/YOLO_plate_dataset
二.图片推理

在 Anaconda Prompt 中，执行命令
yolo detect predict model=runs/detect/train/weights/best.pt source=test_imgs imgsz=640 save=True
三.视频推理
执行命令：
yolo detect predict model=runs/detect/train/weights/best.pt source=test_video.mp4 save=True