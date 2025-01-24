import numpy as np

class Functional(): 
    @staticmethod
    def ReLU(net_input):
        """ReLU激活函数，值域为max(0, z)"""
        return np.maximum(net_input, 0)
    
    @staticmethod
    def ReLU_delta(_input):
        """ReLU导函数"""
        return np.where(_input, 1, 0)
    
    @staticmethod
    def Identical(net_input):
        """恒等函数"""
        return net_input
