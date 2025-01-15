import random
import numpy as np

from models import PointModel
from loss import SquareLoss

seed = 40
random.seed(seed)
np.random.seed(seed)

model = PointModel(lr = 5e-3)
loss = SquareLoss(model)

def loss_callback(predict, target):
    return loss(predict, target)

def train(train_dataset):
    # 迭代训练，用于查看损失函数变化
    for i in range(3):
        for j in range(len(train_dataset)):
            
            features, results = train_dataset[j]
            outputs = model(np.array([features]))
            
            # 手动计算损失函数 
            loss = loss_callback(outputs, results)

            # 反向传播-计算梯度
            loss.backward()
            
            print(f'epoch:{i}; batch_size:{j}; loss:{loss.loss}; loss_error:{loss.loss_error}')
            #print('')


def test(test_dataset):

    for i in range(len(test_dataset)):
        features, result = test_dataset[i]
        output = model(np.array([features]))

        print(output)
        print()


if __name__ == '__main__':
    # 数据集, 来验证模型是否可以学习到权重接近 2 的映射，假设神经元函数中的仿射函数为y = 2x 的线性函数。
    
    # train_dataset 数组中的元素是两元数组，第一个表示特征值，也可以理解为 x；第二个是对应的 y;
    # 若在数据集中加入 [[0.0], 0.0]？？？
    train_dataset = [
        [[2.0], 4.0], [[1.0], 2.0], [[3.0], 6.0], [[0.5], 1.0], [[1.25], 2.5], 
        [[5.0], 10.0], [[1.5], 3.0], [[2.5], 5.0], [[4.0], 8.0], [[3.5], 7.0]
    ]

    test_dataset = [
        [[1.25], 2.5], [[2.0], 4.0], [[4.0], 8.0]
    ]
    

    # 这个训练的很像是回归模型呀，线性的回归模型？？因为预测的是实数
    train(train_dataset)
    test(test_dataset)


