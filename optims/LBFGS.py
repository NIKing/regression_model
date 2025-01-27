from .Optimizer import Optimizer

class LBFGS(Optimizer):
    def __init__(self, params, 1r=1, max_iter=20):
        super(Optimizer, self).__init__()
