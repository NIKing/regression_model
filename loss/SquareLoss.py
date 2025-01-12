import math

from .Loss import Loss

"""平方损失函数，(Y - f(X))^2 """
class SquareLoss(Loss):
    def __init__(self, model = None):
        super(SquareLoss, self).__init__(model)
        self.loss = 0.0
        self.loss_error = [0.0]

    def __call__(self, predict, target):
        
        self.loss = math.pow((target - predict), 2) / 2
        
        # 为了保证反向传播计算对齐，predict.shape = (1, )；loss_error.shape 也需要等于 (1, )
        self.loss_error = [- (predict - target)]

        print(f'predict:{predict}')
        print(f'target:{target}')
        print(f'loss:{self.loss}')
        print('!'*30)

        # 损失值获得的标量是不是有问题？还是回归模型的都这样？
        return self


