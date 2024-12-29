import math
import random

import numpy as np

from .Model import Model

class activation_map(): 
    
    @staticmethod
    def ReLU(net_input):
        """RuLU激活函数，值域为max(0, z)"""
        return np.maximum(net_input, 0)

class HiddenLayer():
    def __init__(self, activation = 'ReLU'):

        # 定义神经元为6; 
        self.hidden_size = 6

        # 输入参数为1;
        self.input_dim = 1

        # 根据输入向量的数量(N)，初始化神经元矩阵(N * 6)，并进行 He 初始化
        # He初始化，对ReLU激活函数等正向激活情况做优化，使用N(0, 2/n)的分布, N 表示正太分布的意思；n 表示神经元数量
        #self.unit_matrix = [[random.random() * 2 / self.hidden_size] * len(features) for i in range(self.input_dim)] 

        # 虽然，理论上权重矩阵的应该由【输入向量数（输入特征维度）* 神经元数量】组成
        # 但是，输入的向量不能在初始化的时候获取到（如果在执行的时候初始化，每执行一次训练都会重置权重），因此需要固定输入特征维度
        # 在bert模型中，hidden_size = 768, [batch_size, input_dim] * [input_dim, output_dim]
        self.unit_matrix = [[random.random() * 2 / self.hidden_size] * self.hidden_size for i in range(self.input_dim)]

    def __call__(self, features):
        return self.activation_fn(self.core_fn(features))
    
    def core_fn(self, features):
        #print(self.unit_matrix)
        #print(features)

        """仿射函数（偏置等于 0，现在的仿射函数就是线性函数, y = w*x ）"""
        return np.dot(features, self.unit_matrix)

    def activation_fn(self, net_input):
        """RuLU激活函数，值域为max(0, z)"""
        return np.maximum(net_input, 0)


class OutputLayer():
    def __init__(self):
        # 定义神经元为1；
        self.hidden_size = 1

        # 输入参数为6;
        self.input_dim = 6
        
        self.unit_matrix = [[random.random() * 2 / self.input_dim] * self.hidden_size for i in range(self.input_dim)]

    def __call__(self, features):
        return self.activation_fn(self.core_fn(features))
    
    def core_fn(self, features):
        #print(self.unit_matrix)
        #print(features)

        """仿射函数（偏置等于 0，现在的仿射函数就是线性函数, y = w*x ）"""
        return np.dot(features, self.unit_matrix)

    def activation_fn(self, net_input):
        return net_input

class PointModel(Model):
    def __init__(self, lr = 0.0):
        super(PointModel, self).__init__()
     
        self.learning_rate = lr

        self.h1 = HiddenLayer()
        self.h2 = OutputLayer()
        
        self.layers = {
            'h1': self.h1,
            'h2': self.h2
        }

        self.parameters = {
            'weight': {
                'h1': self.h1.unit_matrix,
                'h2': self.h2.unit_matrix
            }
        }

    def forward(self, features):

        # 第一层
        h_1 = self.h1(features)
        #print(f'h_1={h_1}')

        h_2 = self.h2(h_1)
        #print(f'h_2={h_2}')

        self.inputs = {'h1': h_1, 'h2': h_2}

        return h_2[0]

