# Speculative decoding
- 一种用于加速大语言模型推理的高效技术，核心在于不是所有的token生成难度都一样
- 因此我们可以使用一个参数量极小，推理速度极快的模型先快速猜出来token然后由大冒险一次性进行并行验证
- 传统的前向传播是需要加载GPU K次的，导致大部分时间都在搬运数据，但是speculative decoding由于有草稿在前面提供了候选的token，能够一次传播同时计算所有位置的概率矩阵，GPU只用跑一次
- 是否接受草稿模型给出的token：
1. 大模型对该token的偏好高于小模型->直接接受该token
2. 大模型对该 token 的偏好较低，但仍以(大模型接受概率/小模型接受概率)的概率接受该 token
3. 若未能通过 Case 2 的判定，则拒绝该 token并放弃后续所有草稿，重新采样一个新的 token作为替代

# RLHF训练(offline和online)
- online是指智能体在训练过程中能够实时与环境交互并采集新数据
- offline是指训练完全基于预先记录的数据集，无法探索/预测动作
- 两种引擎网络架构
1. rollout generator：基于当前策略生成文本回答
2. 接收采样数据然后计算出梯度，将模型参数更新到最新的策略
- 由于推理和训练交替进行，导致巨大的GPU算力浪费
- 解决方法如下：
1. 将生成与训练在时间轴上重叠（Overlap）。在 GPU 训练上一步梯度的同时，推理引擎已使用上一代权重（如 $\theta_t$）开始生成下一批数据，代价为引用off-policy偏离
2. 允许生成引擎在文本生成中途接收并加载最新权重，可以短暂的暂停，加载新的权重，继续生成当前序列

# on-policy distillation
- 以学生模型为中心的大模型，与传统教师强行给学生灌输答案的off-policy不一样
- 传统的蒸馏：有真实的标注/教师模型生成的序列，学生模型只能被动拟合教师在这些特定上下文下的分布
- on-policy distillation：教师模型只是指导者，对学生自己走出的文本路径给出概率评估，训练学生模型学会从自己的错误中纠正

# RoPE
- 让第 1 个词(位置 1)：指针旋转 $1 \times \theta$ 度
- 让第 2 个词(位置 2)：指针旋转 $2 \times \theta$ 度
- 让第 5 个词(位置 5)：指针旋转 $5 \times \theta$ 度
- 假设原本q的初始语义角度是 $\phi_q$，Key 向量 $k$ 的初始语义角度是 $\phi_k$
- 最后的角度差：$$\text{Total Angle} = (\phi_q + m\theta) - (\phi_k + n\theta) = \underbrace{(\phi_q - \phi_k)}_{\text{语义角度差}} + \underbrace{(m - n)\theta}_{\text{位置角度差}}$$
- 加入cos后：$$\cos\Big( (\phi_q - \phi_k) + (m - n)\theta \Big) = \underbrace{\cos(\phi_q - \phi_k)}_{\text{语义关联度}} \cdot \cos((m-n)\theta) - \underbrace{\sin(\phi_q - \phi_k)}_{\text{语义正交度}} \cdot \sin((m-n)\theta)$$
- 不论它们出现在第 10 和 11 个字，还是第 100 和 101 个字，语义角度差 $(\phi_q - \phi_k)$ 始终保持不变，对距离都是1
- 旋转的好处：不会改变长度，点积自动提取角度差(点积过后绝对位置会变成相对距离)