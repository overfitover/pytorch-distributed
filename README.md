# distributed-pytorch

modified version of src

python main.py /home/zhangzhaoyu --dist-rank 0

python main.py /home/zhangzhaoyu --dist-rank 1

## 初始化方法可以执行,三种初始化方式在init_examples里面.

## simple_demo里面是一些小例子,　可以跑起来.　
但是给我的感觉是几个独立的model没什么联系.
每组的loss都相同.理论上如果完全独立的化loss应该随机的.是不是在一定程度上说明他们之间有联系.

## main.py 函数
跑别人的demo的时候init有问题.总是在init卡着.
暂时不想弄了,发现一个更好用的东西.

### 参考
https://github.com/narumiruna/pytorch-distributed-example
https://github.com/uber/horovod

