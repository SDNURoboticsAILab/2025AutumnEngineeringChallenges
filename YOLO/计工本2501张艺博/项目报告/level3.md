一.手动划分数据集
创建子文件夹：打开 D:/YOLO_plate_dataset，在 images 和 labels 文件夹下 分别新建 3 个子文件夹：train、val、teest,之后分配文件。
二.创建「data.yaml」

打开 D:/YOLO_plate_datase,新建data.yaml

path: C:\YOLO_plate_dataset

train: images\train
val: images\val
test: images\test

nc: 1

names: ['license_plate']     
三.启动模型训练
切换到数据集路径：在 Anaconda Prompt 中执行命令，进入 D:/YOLO_plate_dataset
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8