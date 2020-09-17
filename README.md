# achieve-DenseNet

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)

### DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。

#### 1.在跨层连接上，不同于ResNet中将输入与输出相加，DenseNet在通道维上连结输入与输出。
#### 2.DenseNet的主要构建模块是稠密块和过渡层。
