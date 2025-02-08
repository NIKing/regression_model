class Model():
    def __init__(self):
        self.layers = {}
        self.training = True

        self.learning_reate = 1e-3

    def __call__(self, input_ids):
        return self.forward(input_ids)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, input_ids):
        pass
    
    def get_parameters(self):
        return self.parameters

    def update_parameters(self, layer_num, weights):
        """更新权重参数"""
        self.layers[layer_num].weight_matrix = weights

    def update_gamma(self, layer_num, gamma):
        """更新缩放因子"""
        self.layers[layer_num].gamma = gamma
