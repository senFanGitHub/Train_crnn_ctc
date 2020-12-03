## requirement
mxnet-cu
opencv-python
...


## info
本项目以demo验证码数据为例，展示文本行识别的crnn算法训练。

可用于训练文本行识别 ，包括但不限于普通定长、不定长字符验证码,通用ocr文本行。
项目代码主要包含3个类：

crnn_data：根据标注数据生成crnn训练数据流，标注格式按照图像命名方式标注。（忽略重名图像）

crnn_trainer：crnn训练类

crnn_infer：加载crnn_trainer，用于部署的推理类

训练采用cos学习率策略，默认设置为累记训10个epoch验证集仍无进步则终止训练。
更多参数参见python Train_crnn.py -h

## 训练

`python Train_crnn.py   -train_root=Demo_train  -test_root=Demo_test  -max_len=4 -model_dir='model_test' -backbone="ResNetV2_small"`

