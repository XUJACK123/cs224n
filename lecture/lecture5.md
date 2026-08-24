# Neural Machine Translation
- 目的是计算生成整个句子的概率，把是有词语生成的概率相乘(在已知前面内容的情况下一个词的概率)
- encoder:将输入的东西变为向量，最后传递给decoder
- decoder:基于encoder的上下文信息，预测句子是什么样子
- 注意：如果中间decoder猜错了，会直接给准确答案让它去预测下一个词的概率，避免一步错步步错

## Neural Machine Translation的问题：
- 在传统的Seq2Seq中，真正传给Decoder的只有最后的向量，也就是说一个向量要包含前面的所有语义信息，这会导致RNN的老毛病，容易遗忘前面的信息
- decoder会在后面生成中对于原本encoder给出的信息比重越来越少，只靠前后文来推断

# attention
1. attention score:将decoder的当前状态与encoder的每个向量(每个词的信息)进行对比
2. attention distribution:对所有encoder中的向量打分，并且把重要的向量提高关注度
3. context vector:把encoder中所有vector进行加权求和，变为一个新的向量(可以理解为这个向量就是encoder为这个decoder的词的专属向量)
4. predict output:将decoder自己的状态(decoder中的前后文)，与前面的向量进行拼接，算出词表中最大的概率，选出最后的词
- 使得decoder不再依赖单一的固定向量，随时可以读取encoder中的东西
- 在decoder和encoder中建立了直接的连接，加快反向传播，没有attention的时候，需要由时间轴反向回去到encoder的状态来进行更新，容易有梯度消失的问题，有attention后可以直接回到encoder因为decoder连接了encoder

# self attention
- RNN的问题：1.必须等前面的词算完才能算后面的词，2.即使有attention的辅助，也依赖网络来传送信息
- QKV机制：
    - Q:当前词想寻找什么信息
    - K:当前词能提供什么特征标识(内容大纲，作为快速匹配)
    - V:当前词实际蕴含的语义内容(真正的内容)
    1. 先把每一个词的Q与所有词的K做点积，一次性算出全句所有词两两之间的关联度矩阵
    2. 归一化
    3. 用前面的结果按比例融合进这个词本身
    - 最后的输出变为每一个词词自身+全局上下文信息，使得每个词都融合了全文信息
- 缺陷：缺乏位置信息
- 解决方法：把每个词的向量加入位置信息，现在多使用RoPE，意思是高数两个词的相对距离，而不是绝对距离
- 缺陷：容易偷看后面的目标词(self attention默认全局可看)
- 解决方法：把未来位置设为负无穷

# transformer
- Transformer Decoder：用来生成式的任务或者是语言模型
- Transformer Encoder：包含无掩码的 Multi-Head Attention，用于文本理解与特征提取

## Multi-Head Attention
- 由于单词在句子中需要关注不同的东西(例如：要观察语义，也要观察语法)，设置多头的的attention，每个attention都有自己的QKV

## Scaled Dot-Product Attention
- 将注意力得分除以 $\sqrt{d/h}$(h为多头注意力的数量)，即为单个注意力头的维度，因为随着特征维度的增大，向量点积的数值会变得非常大，在经过softmax的时候的时候容易趋近于0/1