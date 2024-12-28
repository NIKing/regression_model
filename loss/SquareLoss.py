import math

from .Loss import Loss

"""平方损失函数，(Y - f(X))^2 """
class SquareLoss(Loss):
    def __init__(self, model = None):
        super(SquareLoss, self).__init__(model)

        self.loss = 0.0

    def __call__(self, predict, target):
        self.loss = math.pow((target - predict), 2) / 2
        print(f'predict:{predict}')
        print(f'target:{target}')
        print(f'loss:{self.loss}')
        print()
        return self


