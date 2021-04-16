# AIProject2
清华大学人工智能基础课程大作业2——图像分类

环境：
Windows 10
python 3.7.0

训练代码：
```bash
python train.py ./options/***.json
```

双模型训练代码：
```bash
python cross_train.py ./options/***.json
```

双模型测试代码：
```bash
python cross_test.py /file/name/in/logs  epoch_number
```
例如：
```bash
python cross_test.py class100 200
```
