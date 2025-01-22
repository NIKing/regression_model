class Model():
    def __init__(self):
        self.layers = {}
        self.learning_reate = 1e-3

    def __call__(self, input_ids):
        return self.forward(input_ids)

    def forward(self, input_ids):
        pass

    def get_parameters(self):
        return self.parameters

    def update_parameters(self, layer_num, weights):
        """更新权重参数"""
        
        #print('----')
        #print(self.layers[layer_num].weight_matrix.shape)
        #print(layer_num, weights.shape)
        #output_dim, input_dim = self.layers[layer_num].output_dim, self.layers[layer_num].input_dim
        #print(weights.view(output_dim, input_dim))
        #print('+++++')

        self.layers[layer_num].weight_matrix = weights
