# attention_module
分为**通道注意力**和**空间注意力**。前者关注同一位置处哪种特征重要（一个通道提取一种特征，如轮廓，明暗等），后者关注特征图中哪个位置更重要。
越重要的地方权值越大。

## SEnet
可以实现通道注意力机制。
![SEnet](https://img-blog.csdnimg.cn/20201124130209827.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)
实现方式：
1. 拿到特征层，进行全局平均池化，得到维度为[1, 1, c]的特征向量
2. 对特征向量进行两次全连接，先缩减通道数，再放回原来通道数
3. 对处理后的特征向量进行sigmoid激活，使得每个值都位于(0,1)，可作为权值
4. 原特征层中每一通道所有特征点都乘以对应的权值，进行注意力

## CBAM
可以实现通道注意力和空间注意力
![CBAM1](https://img-blog.csdnimg.cn/20201124133821606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)
![CBAM2](https://img-blog.csdnimg.cn/20201124134115869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)
实现方式：
整体上看，对于输入特征层，先做**通道注意力**，在做**空间注意力**
- 通道注意力
    1. 对输入特征层，分别做全局平均池化和全局最大池化，维度为[1, 1, C]
    2. 利用共享全连接层，做两次全连接（先降维再升维），维度为[1, 1, C]
    3. 两个结果相加，并进行sigmoid激活，获得每个通道的权值，维度为[1, 1, C]
    4. 特征层中每一通道所有特征点都乘以对应的权值，进行注意力
- 空间注意力 
    1. 对输入特征层，在每个特征点的通道上分别取最大值和平均值，维度为[W, H, 1]
    2. 两个结果堆叠，维度为[W, H, 2]
    3. 利用通道为1的卷积调整通道数，维度为[W, H, 1]
    4. 进行sigmoid激活，获得每一个特征点的权值
    5. 特征层中每一特征点的所有通道都乘以对应的权值，进行注意力

## ECAnet
可实现通道注意力
