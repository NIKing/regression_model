class Model():
    def __init__(self):
        self.layers = {}
        self.parameters = {'weight': {}}
        self.inputs = {}

    def __call__(self, input_ids):
        return self.forward(input_ids)

    def forward(self, input_ids):
        pass

    def get_parameters(self):
        return self.parameters
