<style>
</style>

# 基于 YOLO 的车牌号识别系统学习文档

## 一．环境搭建：

### 1.从哔哩哔哩搜索yolo11安装视频及安装文档，进行认真学习，并按照视频进行了安装，由于初学，安装均不成功，通过多次摸索，最终主要参考以下链接完成：

`https://www.bilibili.com/opus/1021487181719404550`，YOLOv11(Ultralytics)环境配置，适合0基础纯小白，手把手教 - 哔哩哔哩

Anaconda3文件包下载网站：`https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D`

Pycharm安装包下载

`https://www.jetbrains.com/zh-cn/pycharm/download/?section=windows`

两个文件选择合适版本进行安装。中间由于版本问题，进行了多次下载安装。

### 2.创建yolo11环境

由于我所使用的电脑为Intel核显，没有独立显卡无法使用GPU进行训练，需用CPU进行训练。

打开anaconda prompt，运行以下命令创建yolo11环境：

```python
conda create -n yolov11 python=3.10
```

创建完成后，输入

```python
conda activate yolov11
```

进入yolo环境。

### 3. 安装pytorch

```python
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 4. 安装ultralytics库

```python
pip install ultralytics
```

### 5.下载YOLOv11（ultralytics）源码，

由于哔哩哔哩搜索的学习文档中给出的网址一直无法登录，重新学习实验室下发的链接中文档，找到下载链接，成功下载ultralytics源码见下图。

![图片alt](C:\Users\honor\Desktop\图片1.png "图片title")

下载预训练权重文件yolo11n、yolo11s、yolo11m，下载界面如下：

![图片2.png](C:\Users\honor\Desktop\图片\图片2.png)

### 6. pycharm导入环境

导入界面：

![图片3.png](C:\Users\honor\Desktop\图片\图片3.png)

## 二．下载和标注数集

### 1.按照文档内容，下载数据集。

网址：`https://pan.baidu.com/s/1-WmsMt7Zzx3jmLHVEYpdaw`

提取码: 2111

### 2.学习文档中ccpd数据集处理，并从哔哩哔哩搜索LabelStudio安装教程及数据标记教程。

参考文档及视频：
`https://blog.csdn.net/m0_72915515/article/details/134248502#:~:text=%E9%87%87%E7%94%A8Anaconda%E5%88%9B%E5%BB%BA%E5%90%8D%E4%B8%BAlabel_sudio%E7%9A%84%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83`，标注工具——Label Studio安装与简单使用-CSDN博客

`https://www.bilibili.com/video/BV1Ht4y1d7Yr/?spm_id_from=333.337.search-card.all.click&vd_source=0e4eaed4406fd009f7a2bd20660a23a4 `，数据标注的方法 制作自己的数据 labelstudio的使用_哔哩哔哩_bilibili

`https://www.bilibili.com/video/BV1K3zwYQEgT/?spm_id_from=333.337.search-card.all.click&vd_source=0e4eaed4406fd009f7a2bd20660a23a4`， label studio使用教程_哔哩哔哩_bilibili

打开anaconda prompt，运行以下命令：

```python
pip install Label-Studio
```

安装成功后，运行Label-Studio start，进入web页面进行数据集标注。

### 3.在导入本地图片过程中，由于使用了中文翻译版，一直提示“要插入新节点的节点不是该节点的子节点”错误

查找多个文档未解决，最后重新**登陆英文版界面**，成功将数据导入，并完成标注，进行导出，一共完成113张车牌数据标注。

## 三、划分数据集，配置并训练模型

### 1.数据集划分：

根据文档，将数据集目录在划分为**train、val、test**三个目录。

### 2.data.yaml文档配置及模型训练

第一便训练提示如下错误，根据错误提示先进行网络配置排查，最终**调整网络设置**进行解决：

![图片4.png](C:\Users\honor\Desktop\图片\图片4.png)

第二遍训练提示如下错误，经过检查问data,**yaml文件配置中文件名输入错误**导致：

![图片5.png](C:\Users\honor\Desktop\图片\图片5.png)

第三遍训练一直如下错误，经过多次多次排查，最终发现**标签txt文件导出为空**导致，重新使用label studio进行标注后导出文件，为便于排查，本次只使用6张车牌，结果成功导出：

![图片6.png](C:\Users\honor\Desktop\图片\图片6.png)

所有文件修改后进行再次进行训练，成功：

![图片7.png](C:\Users\honor\Desktop\图片\图片7.png)

**Data.yaml文档代码**

```python
path: d:/test
train: images/train #训练集
val: images/val #验证集
test: images/test #测试集
nc: 1
names: ['car']
```

**test.py程序代码**：

```python
from ultralytics import YOLO

# 加载 YOLOv11n 模型
model = YOLO('yolo11n.pt')  # 使用预训练模型

# 开始训练
results = model.train(
    data='data.yaml',     # 数据集配置文件
    epochs=100,              # 训练轮数
    imgsz=640,               # 图像尺寸
    batch=16,                # 批次大小
    device='cpu',                # 设备使用cpu
    name='yolo11n_custom'    # 实验名称，结果保存在 runs/detect/yolo11n_custom/
)

# 可选：评估模型
metrics = model.val()
print("mAP50-95:", metrics.box.map)
```

### 3.训练结束输出结果：

![图片8.png](C:\Users\honor\Desktop\图片\图片8.png)![图片9.png](C:\Users\honor\Desktop\图片\图片9.png)

<img title="" src="file:///C:/Users/honor/Desktop/图片/图片10.png" alt="图片10.png" width="295"><img src="file:///C:/Users/honor/Desktop/图片/图片11.png" title="" alt="图片11.png" width="293">

<img src="file:///C:/Users/honor/Desktop/图片/图片12.png" title="" alt="图片12.png" width="241">

<img src="file:///C:/Users/honor/Desktop/图片/图片13.png" title="" alt="图片13.png" width="249">

<img src="file:///C:/Users/honor/Desktop/图片/图片14.png" title="" alt="图片14.png" width="261">

<img src="file:///C:/Users/honor/Desktop/图片/图片15.png" title="" alt="图片15.png" width="276">

## 四、使用训练好的模型进行推理

### 1.图片推理

新建xunlian程序，代码如下（从文档中复制后修改为实际文件路径等参数）：

```python
import cv2
# 引入YOLO模型
from ultralytics import YOLO
# 打开图像
img_path = "d:/xunlian/img.jpg"  # 这里修改你图像保存路径
# 打开图像
img = cv2.imread(filename=img_path)
# 加载模型
model = YOLO(model="yolo11n.pt")  # 这里修改你图像保存路径
# 正向推理
res = model(img)
# 绘制推理结果
annotated_img = res[0].plot()
# 显示图像
cv2.imshow(winname="YOLO11", mat=annotated_img)
# 等待时间
cv2.waitKey(delay=10000)
# 绘制推理结果
cv2.imwrite(filename="jieguo.jpeg", img=annotated_img)
```

显示结果如下，图片成功。

![图片16.png](C:\Users\honor\Desktop\图片\图片16.png)

### 2.视频推理

新建shipinxunlian程序，代码如下（从文档中复制后修改为实际文件路径等参数）：

```python
import cv2
from ultralytics import YOLO
# 加载模型
model = YOLO(model="yolo11x.pt")
# 视频文件
video_path = "d:/xunlian/chepai.mp4"
# 打开视频
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    # 获取图像
    res, frame = cap.read()
    # 如果读取成功
    if res:
        # 正向推理
        results = model(frame)
        # 绘制结果
        annotated_frame = results[0].plot()
        # 显示图像
        cv2.imshow(winname="YOLO11", mat=annotated_frame)
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
    else:
        break
# 释放链接
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()
```

显示结果如下，视频成功

![图片17.png](C:\Users\honor\Desktop\图片\图片17.png)
