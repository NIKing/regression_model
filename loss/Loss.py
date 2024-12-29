class Loss():
    def __init__(self, model = None):
        self.model = model

    def __call__(self, predict, target):
        return self


    def calculate_layer_error(self, layer, next_layer_error, next_layer_weight):
        """
        每一层的误差，是由损失函数对净输入的梯度计算
        不过，根据链式规则，当前层的“误差”是由下一层的“误差”和“权重”，再经过对当前层激活函数进行求导，得到的梯度
        -param layer 当前网络层
        -param next_layer_error 下一层“误差”
        -param next_layer_weight 下一层“权重”
        """
        #return self.

    def backward(self):
        """反向传播的计算方式是通过损失函数/权重，得到梯度值"""

        next_layer_error = 0.0
        next_layer_weight = []
        
        weight_items = list(self.model.parameters['weight'].values())
        input_items = list(self.model.inputs.items())
        layer_items = list(self.model.layers.values())
        
        print(input_items)
        # 从后向前计算梯度
        for i in range(len(input_items) - 1, 0, -1):
            
            # 首先计算【输出层】的误差, 这个在外部已经计算过，直接拿过来用
            if i == len(input_items) - 1:
                layer_error = self.loss 
            else:
                layer_error = self.calculate_layer_error(layer_items[i], next_layer_error, next_layer_weight)
           
            # 获取当前层和上一层的输出结果
            current_layer_num, current_input = input_items[i]
            last_layer_num, last_input = input_items[i - 1]
            
            # 计算梯度：当前层误差值 * 上一层的输出
            layer_gradient = layer_error * last_input
            print(f'第{i}层的权重:', weight_items[i])
            print(f'第{i}层的梯度:', layer_gradient)

            # 更新参数
            current_weight = weight_items[i]
            new_weight = current_weight - [self.model.learning_rate * gradient for gradient in layer_gradient]
            
            self.model.update_parameters(current_layer_num, new_weight)
            
            # 记录信息，用于计算上一层的误差
            next_layer_error = layer_error
            next_layer_weight = current_weight
