## level5：结合 OCR 进行车牌字符识别

这是一个充满曲折的level5，报错报错还是报错。

### 1.安装OCR库

我们需要在虚拟环境中安装PaddlePaddle 和 PaddleOCR。

由于我的电脑有NVIDIA显卡且配置了CUDA，我想下载gpu版本。ai让我运行``pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple``命令完成下载。

butbutbutbut出现报错，pip找不到一个能匹配我当前环境的安装包。我于是转战官网……

（此处省略一个漫长的下载过程……

接下来是paddleocr

butbutbutbutbutbutbutbutbutbut安装出现了一系列的问题，报错报错还是报错。

报错信息：OSError: [WinError 127] 找不到指定的程序。 

```
Error loading "D:\anaconda\envs\yolov8\lib\site-packages\paddle..\nvidia\cudnn\bin\cudnn_cnn64_9.dll" or one of its dependencies.
```

我在官方文档中看到

> 通过以上方式安装的 PaddlePaddle 在 Windows 操作系统下无法正常支持 NVIDIA 50 系显卡。因此，我们提供了专门适配该硬件环境的 PaddlePaddle 安装包。请根据您的 Python 版本选择对应的 wheel 文件进行安装。

那么好，运行官方指令

```bash
# python 3.9
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp39-cp39-win_amd64.whl
```

尝试安装后还是出现了同样的问题，看了下github issue，感觉应该是官方还没有适配50系显卡的问题。

只好换成cpu版本的下载，下载完成后，在终端中运行ai提供的代码。

