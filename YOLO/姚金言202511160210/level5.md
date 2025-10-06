## level5：结合 OCR 进行车牌字符识别

这是一个充满曲折的level5，报错报错还是报错。

### 1.安装OCR库

我们需要在虚拟环境中安装PaddlePaddle 和 PaddleOCR。

由于我的电脑有NVIDIA显卡且配置了CUDA，我想下载gpu版本。ai让我运行``pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple``命令完成下载。

butbutbutbut出现报错，pip找不到一个能匹配我当前环境的安装包。我于是转战官网……

（此处省略一个漫长的下载过程……

接下来是paddleocr

butbutbutbutbutbutbutbutbutbut安装出现了一系列的问题，报错报错还是报错。

根据ai的指示一个个排查报错原因，最终发现paddleocr尚未推出支持英伟达50系显卡的版本。

只好换成cpu版本的下载，下载完成后，在终端中运行ai提供的代码。