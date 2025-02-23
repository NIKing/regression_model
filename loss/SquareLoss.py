import math
import numpy as np

from loss.Loss import Loss

"""平方损失函数，(Y - f(X))^2 """
class SquareLoss(Loss):
    def __init__(self, model = None):
        super(SquareLoss, self).__init__(model)

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0
    
    def item(self):
        return self.loss

    def __call__(self, predict, target):
        #print(f'predict:{predict}')
        #print(f'target:{target}')

        self.batch_size = predict.shape[0]
        
        # 损失值，衡量模型预测结果好坏，不参与反向传播
        # 0.5 这个系数，本身就是罚项，有助于损失值不上溢，但不是必须的
        self.loss = np.mean(0.5 * (predict - target) ** 2)
        
        # 为了让误差参与反向传播过程的计算，必须保证 loss_error.shape = (batch_size, in_features) 也就是矩阵形式
        # 貌似一切张量的计算，都是以二维矩阵形式计算，而不是更高维度或更低维度 -- 2025年1月12日
        # 对于上面“一切张量的计算，都是以二维形式来处理”这个说法做出更正，在"全连接层"往往是以2维方式计算，但是在CNN, RNN, Transformer和Attention等结构中并非如此，实际张量维度会更高。
        # 而现在是手搓基础模型，还是以全连接层为主，因此，整个模型从输入到输出的维度都保持在2维即可
        self.loss_error = predict - target

        # 损失值获得的标量是不是有问题？还是回归模型的都这样？
        return self


