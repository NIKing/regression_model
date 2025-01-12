import numpy as np

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
        return layer.delta_fn(layer.net_input) * np.dot(next_layer_error, next_layer_weight.T)

    def backward(self):
        """反向传播的计算方式是通过损失函数/权重，得到梯度值"""

        next_layer_error = 0.0
        next_layer_weight = []
        
        weight_values = list(self.model.parameters['weight'].values())
        layer_items = list(self.model.layers.items())
        
        # 从后向前计算梯度
        for i in range(len(layer_items) - 1, 0, -1):
            layer_number, layer = layer_items[i]
            
            # 首先计算【输出层】的误差, 这个在外部已经计算过，直接拿过来用
            if i == len(layer_items) - 1:
                layer_error = np.array(self.loss_error)
            else:
                # 从下一层网络中，计算出误差
                layer_error = self.calculate_layer_error(layer, next_layer_error, next_layer_weight)
            
            #print('当前层的误差', layer_error, layer_error.shape)

            # 获取上一层的输出结果
            last_input = layer_items[i - 1][1].output
            #print(f'第{i-1}层输出值', last_input.T, last_input.T.shape)
            #print('')

            # 计算梯度：当前层误差值 * 上一层的输出; 需要保证形状对齐; 
            # last_input.T.shape = (in_features, batch_size), layer_error.shape = (batch_size, out_features); 
            # layer_gradient.shape = (in_features, batch_size) * (batch_size, out_features) = (in_features, out_features)
            layer_gradient = np.dot(last_input.T, layer_error)
            current_weight = np.array(weight_values[i])
            #print(f'第{i}层的梯度:', layer_gradient)
            #print(f'第{i}层的权重:', current_weight)

            # 更新参数
            new_weight = current_weight - self.model.learning_rate * layer_gradient

            # 恢复权重形状，保存到网络层
            #print(f'第{i}层的新权重', new_weight)
            
            self.model.update_parameters(layer_number, new_weight)
            
            # 记录当前信息，用于误差传播
            next_layer_error = layer_error
            next_layer_weight = new_weight 

        
                
