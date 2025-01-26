import math
import random

import numpy as np
from functional import Functional

class LinearLayer():
    def __init__(self, input_dim = 6, output_dim= 6, activation = 'ReLU'):

        # 定义神经元为6; 
        self.output_dim = output_dim

        # 输入参数为1;
        self.input_dim = input_dim
        
        self.input = None
        self.net_input = None
        self.output = None

        # 根据输入向量的数量(N)，初始化神经元矩阵(N * 6)，并进行 He 初始化
        # He初始化，对ReLU激活函数等正向激活情况做优化，使用N(0, 2/n)的分布, N 表示正太分布的意思；n 表示神经元数量
        #self.unit_matrix = [[random.random() * 2 / self.output_dim] * len(features) for i in range(self.input_dim)] 

        # 虽然，理论上权重矩阵的应该由【输入向量数（输入特征维度）* 神经元数量】组成
        # 但是，输入的向量不能在初始化的时候获取到（如果在执行的时候初始化，每执行一次训练都会重置权重），因此需要固定输入特征维度
        # 在bert模型中，hidden_size = 768, [batch_size, input_dim] * [input_dim, output_dim]
        self.weight_matrix = np.array([[random.random() * 2 / self.output_dim] * self.output_dim for i in range(self.input_dim)])
        
        #print(self.input_dim, self.output_dim, self.weight_matrix.shape)
        
        # 定义下激活函数
        self.activation = activation
        self.activation_fn = Functional.Identical
        self.delta_fn = None

        if activation == 'ReLU':
            self.activation_fn = Functional.ReLU
            self.delta_fn = Functional.ReLU_delta

    def __call__(self, features):
        """这是代表神经元函数，每个输入都需要与权重参数发生线性变换，再经过非线性变换，最后输出"""
        #print('&'*30)
        #print(features.shape)
        #print(self.weight_matrix.shape)
        print('输入:', features)
        self.input = features
        self.net_input = self.affine_fn(features)
        self.output = self.activation_fn(self.net_input)
        print('净输入:', self.net_input)
        #print(self.activation)
        #print(self.output.shape)
        #print('--'*30 )

        return self.output
    
    def affine_fn(self, features):
        """仿射函数（当偏置等于 0 时，仿射函数就是线性函数 y = w*x ）"""
        return np.dot(features, self.weight_matrix)
