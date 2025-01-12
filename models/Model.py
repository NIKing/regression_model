class Model():
    def __init__(self):
        self.layers = {}
        self.parameters = {'weight': {}}

        self.learning_reate = 1e-5

    def __call__(self, input_ids):
        return self.forward(input_ids)

    def forward(self, input_ids):
        pass

    def get_parameters(self):
        return self.parameters

    def update_parameters(self, layer_num, weights):
        self.layers[layer_num].weight_matrix = weights
        self.parameters['weight'][layer_num] = weights
