import numpy as np

def maximum():
    """不管参数的顺如何，结果都是一样，会把小于0的数据置换为0"""
    res = np.maximum([1, 3, 5, -1], 0)
    res1 = np.maximum(0, [1, 3, 5, -1])

    print(res1)
    print(res)

maximum()

def test_mean():
    """mean求的平均，会根据传入参数的第一个维度大小来计算"""
    a = np.array([[2], [2]])
    b = np.array([[3], [3]])
    r = np.mean(a - b)
    print(r)

test_mean()
