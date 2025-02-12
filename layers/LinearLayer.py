import math
import random

import numpy as np
from functional import Functional

class LinearLayer():
    def __init__(self, input_dim = 6, output_dim= 6, activation = 'ReLU', is_normal = True):

        # 神经元（输出维度）; 
        self.output_dim = output_dim

        # 特征数量（输入维度）;
        self.input_dim = input_dim
        
        self.input = None
        self.output = None
        
        self.net_input = None
        self.net_input_normal = None

        # 根据输入向量的数量(N)，初始化神经元矩阵(N * 6)，并进行 He 初始化
        # He初始化，对ReLU激活函数等正向激活情况做优化，使用N(0, 2/n)的分布, N 表示正太分布的意思；n 表示神经元数量
        #self.unit_matrix = [[random.random() * 2 / self.output_dim] * len(features) for i in range(self.input_dim)] 

        # 虽然，理论上权重矩阵的应该由【输入向量数（输入特征维度）* 神经元数量】组成
        # 但是，输入的向量不能在初始化的时候获取到（如果在执行的时候初始化，每执行一次训练都会重置权重），因此需要固定输入特征维度
        # 在bert模型中，hidden_size = 768, [batch_size, input_dim] * [input_dim, output_dim]
        self.weight_matrix = np.array([[random.random() * 2 / self.output_dim] * self.output_dim for i in range(self.input_dim)])
        #print(self.input_dim, self.output_dim, self.weight_matrix.shape)
        
        # 归一化的参数
        self.is_normal = is_normal
        self.gamma = np.array([1.0] * self.output_dim)  # 缩放因子    
        self.beta = np.array([0.0] * self.output_dim)   # 偏移因子
        
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
        self.input = features

        print('输入:', features)
        #print('权重：', self.weight_matrix)
        print('样本均值分布:', np.mean(features, axis=1))
        print('样本方差分布:', np.var(features, axis=1))

        # 仿射变换
        self.net_input = self.affine_fn(features)
        print('净输入:', self.net_input)
        
        if self.is_normal:
            # 归一化
            self.net_input_normal = self.standardization(self.net_input)
            print('归一化净输入:', self.net_input_normal)

            #self.net_input_normal = self.rescaling(self.net_input)
            #print('归一化净输入:', self.net_input_normal)
            
            #print(np.mean(self.net_input_normal, axis=1))
            #print(np.var(self.net_input_normal, axis=1))

            # 二次仿射变换
            self.net_input = self.affine_fn_by_normal(self.net_input_normal)
            print('二次仿射变换:', self.net_input)

        else:
            # 虽然输出层没有归一化，但是为了反向传播计算，需要赋值
            self.net_input_normal = self.net_input      

        self.output = self.activation_fn(self.net_input)
        #print(self.output)
        #print('gamma=', self.gamma)
        #print('beta=', self.beta)
        print('')

        return self.output
    
    def affine_fn(self, features):
        """仿射函数（当偏置等于 0 时，仿射函数就是线性函数 y = w*x ）"""
        return np.dot(features, self.weight_matrix)

    def standardization(self, z):
        """
        标准归一化净输入的值
        -param z tensors 净输入
        return tensors
        """
        # 注意，这里是层归一化处理方法，因此需要对每个样本进行求值，而非 np.mean(z)，当作是mini-batch的样本
        mean_value = np.mean(z, axis=1, keepdims=True) # 均值
        std_value = np.std(z, axis=1, keepdims=True)   # 标准差

        #print('mean_value', mean_value)
        #print('std_value', std_value)

        return (z - mean_value) / (std_value + 1e-5)

    def rescaling(self, z):
        """最小-最大缩放归一化"""
        min_value = np.min(z, axis=1, keepdims=True)
        max_value = np.max(z, axis=1, keepdims=True)
        
        #print('min_value', min_value)
        #print('max_value', max_value)
        
        return (z - min_value) / (max_value - min_value + 1e-5)


    def affine_fn_by_normal(self, z):
        """
        层归一化后的仿射变换
        -param z tensor 净输入
        return tensors
        """
        return self.gamma * z + self.beta

