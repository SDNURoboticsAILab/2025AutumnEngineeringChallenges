一.数据集下载与预处理
1.下载 CCPD 数据集.解压到 D:/CCPD_raw
2.合并待标注图片
新建文件夹 all_plate_data,文件夹中的 所有.jpg 图片（忽略视频文件、ultralytics 文件夹等非图片内容）D:/CCPD_raw/yolo/all_plate_data；
二.安装 LabelStudio
pip install label-studio -i https://pypi.tuna.tsinghua.edu.cn/simple
三.完成手动标注
四.导出 YOLO 格式标注
导出标注文件,解压到新建文件夹D:/YOLO_plate_dataset
我自己标注了几十张，但几千张太多了，我不会快捷方法，就找人要了几千张做好的了