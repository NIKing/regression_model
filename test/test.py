import numpy as np

def maximum():
    res = np.maximum([1, 3, 5, -1], 0)
    res1 = np.maximum(0, [1, 3, 5, -1])

    print(res1)
    print(res)

maximum()
