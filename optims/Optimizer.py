
class Optimizer():
    def __init__(self, params = iter, defaults = dict):
        self.params = params
        self.defaults = defaults

    def step(self):
        pass

    def zero_grad(self):
        pass
