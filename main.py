import random
import numpy as np

from models import PointModel
from loss import SquareLoss
from dataloader import DataLoader
#from optim import LBFGS

seed = 40
random.seed(seed)
np.random.seed(seed)

# 效果最好的参数 [8e-3, 55],
# [6e-3, 5]

model = PointModel(6e-3)
loss = SquareLoss(model)
#optim = LBFGS(model.params, lr=1.5e-2)

def loss_callback(predict, target):
    return loss(predict, target)

def train(train_dataset):

    # 迭代训练，用于查看损失函数变化
    for i in range(5):
        
        print('*'*40, f'第{i+1}轮次训练', '*'*40)

        model.train()

        train_data = DataLoader(train_dataset, shuffle=True, batch_size=2)
        iter_data = iter(train_data)
        batch_data = next(iter_data)

        batch_num = 0
        while train_data.is_next():
         
            features, results = zip(*batch_data)

            features = np.array(features, dtype=np.float64)
            results = np.expand_dims(np.array(results, dtype=np.float64), -1)
            #print(features, features.shape)
            #print(results, results.shape)

            # 模型预测
            outputs = model(features)
            
            # 手动计算损失函数 
            loss = loss_callback(outputs, results)

            # 反向传播-计算梯度
            loss.backward()
            
            print(f'epoch:{i + 1}; batch_size:{batch_num}; loss:{loss.loss}; ')
            print('')

            batch_data = next(iter_data)
            batch_num += 1


def test(test_dataset):
    model.eval()

    for i in range(len(test_dataset)):
        features, result = test_dataset[i]
        features = np.array([features], dtype=np.float64)
        
        print(f'Test Epoech:{i}/{len(test_dataset)}')
        output = model(features)
        print(output)
        print()


if __name__ == '__main__':
    # 数据集, 来验证模型是否可以学习到权重接近 2 的映射，假设神经元函数中的仿射函数为y = 2x 的线性函数。
    
    # train_dataset 数组中的元素是两元数组，第一个表示特征值，也可以理解为 x；第二个是对应的 y;
    # 若在数据集中加入 [[0.0], 0.0] 和 [[-1.0], -2.0] 都没有什么特殊的作用

    train_dataset = [
        [[20.0, 1.2], 42.4], [[10.0, 1.0], 22.0], [[30.0, 1.5], 63.0], [[2.5, 10.5], 26.0], [[1.25, 10.0], 22.5],
        [[11.0, 4.0], 30.0], [[1.5, 10.01], 23.02], [[20.5, 0.0], 41.0], [[40.0, 0.1], 80.2], [[3.5, 30], 67.0],
    ]
    
    train_dataset = [
        [[2.0, 1.2], 6.4], [[1.0, 12.0002], 26.0004], [[3.0, 1.5], 9.0], [[10.5, 0.5], 22.0], [[1.25, 1.0], 4.5],
        [[5.0, 4.0], 18.0], [[1.5, 0.01], 3.02], [[2.5, 0.0], 5.0], [[4.0, 0.1], 8.2], [[3.5, 3], 13.0],
    ]
    
    # 添加一些干扰
    #train_dataset = [[[s * 1e-6 for s in samples], target] for samples, target in train_dataset]

    #check_result = [2 * sum(sample) == target for sample, target in train_dataset]
    #print(check_result)

    test_dataset = [
        [[1.25, 0], 2.5], [[2.0, 2.0], 8.0], [[4.0, 1.0], 10.0]
    ]
    

    # 这个训练的很像是回归模型呀，线性的回归模型？？因为预测的是实数
    train(train_dataset)
    
    test(test_dataset)


