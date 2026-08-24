# neural classification(为了区分命名实体/命名实体识别任务)

## 建立classification的任务来区分是否是命名实体
- 最后是使用sigmoid方程映射为概率然后然后判断是否高于阈值决定是不是命名实体

# Backpropagation
- 通过反向传播，雅可比矩阵来算出每个参数的gradient最后统一进行更改
*** 为什么要统一进行更改：主要就是避免大量的重复计算，反向传播的前半段都是一样的(都得经过sigmoid，output的gradient decent)因此使用：$\boldsymbol{\delta} = \frac{\partial s}{\partial \boldsymbol{z}}$ 来表达这些一样的东西

## 关于雅可比矩阵
- 注意：是有两个雅可比矩阵的个是逐元素激活函数的雅可比($\boldsymbol{h} = f(\boldsymbol{z})$)，一个是线性全连接层的雅可比($\boldsymbol{z} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}$)
- 对于线性全连接层的雅可比：
    - 如果是将n维的向量映射到m维(n个输入，m个输出)，矩阵的维度恒为m x n(因为得满足矩阵乘法)
- 对于逐元素激活函数的雅可比：
    - 因为激活函数是逐元素独立作用的，激活函数的雅可比矩阵是一个对角矩阵